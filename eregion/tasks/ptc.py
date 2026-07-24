from copy import deepcopy
import pandas as pd
from typing import Optional, Any, Callable, Generator
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from functools import partial, wraps
from itertools import combinations
from pydantic import field_validator
import os
import json
import warnings
warnings.filterwarnings("ignore")

from utils import slice_data, save_ptc_table_fits, load_ptc_table_fits
from datamodels import DetImage, CCDOutput, ImageBundle, TaskResult
from tasks import LazyTask
from core.image_stats import *
from core.image_operations import sigma_clip_image
from core.welch2d import welch2d

##################### PTC task ############################
class PTCResult(TaskResult):
    ptc_table: pd.DataFrame
    diff_images: ImageBundle

    @field_validator("diff_images", mode="before")
    @classmethod
    def parse_diff_images(cls, inp):
        if not isinstance(inp, ImageBundle):
            return ImageBundle(inp)
        return inp

    def save(self, filepath: str, save_diffs=False, **kwargs) -> None:
        os.makedirs(filepath, exist_ok=True)
        if save_diffs:
            self.diff_images.save(os.path.join(filepath, "diff_images"), **kwargs)
        save_ptc_table_fits(self.ptc_table, os.path.join(filepath, "ptc_table.fits"))
        super().save(filepath)

    @classmethod
    def load(cls, filepath: str, load_diffs=False):
        if load_diffs:
            diff_images = ImageBundle.load(os.path.join(filepath, "diff_images"))
        else:
            diff_images = ImageBundle()
        ptc_table = load_ptc_table_fits(os.path.join(filepath, "ptc_table.fits"))
        with open(os.path.join(filepath, f"{cls.__name__}_metadata.json"), "r") as f:
            metadata = json.load(f)
        return cls(ptc_table=ptc_table, diff_images=diff_images, **metadata)

class PTC(LazyTask):
    task_result = PTCResult

    def __init__(self,
                 psd_size: int = 9,
                 exptime_key: str = 'exptime',
                 name: str='PTC',
                 **kwargs
                 ):
        """
        Run photon transfer curve (PTC) analysis for input pairs of flat images.
        :param psd_size: Optional[int]
            Size of power spectral density array, by default 9.
        :param exptime_key: str
            Key/Column in the image_type attribute of DetImage that contains the exposure time for that image, by default 'exptime'.
        :param name: Optional[str]
            Name of the task. Default is 'PTC'.

        kwargs:

        - do_sigma_clip : bool
            Whether to perform sigma clipping on differenced output data (only image region)
        - sigma_clip_args: dict[str, Any]
            Additional arguments to pass to the sigma_clip function, e.g., {'sigma': 5.0, 'grow': 10}.
        - welch2d_kwargs : dict[str, Any]
            Additional arguments to pass to the welch2d function,
        """
        super().__init__(name=name, **kwargs)
        self.PSD_size = psd_size
        self.exptime_key = exptime_key

    def lazy_run(self, images: ImageBundle | list[DetImage], **kwargs) -> Generator[PTCResult, None, None]:
        images = images if isinstance(images, ImageBundle) else ImageBundle(images)

        expkey = self.exptime_key
        exptimes = images.list[expkey].unique()
        det_ids = images.list["det_id"].unique()

        results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._process_flat_group)(images.filter(f'{expkey} == {exptime} & det_id == "{det_id}"'))
                    for exptime, det_id in zip(exptimes, det_ids)
        )

        stats, diff_images = [], []
        for result in results:
            stats.extend(result[0])
            diff_images.extend(result[1])
        statsdf = pd.DataFrame(stats)

        self.logger.info(f"Processed {len(images)} flats, upto exptime {max(exptimes)}")
        yield self.task_result(ptc_table=statsdf, diff_images=diff_images)

    # @wraps(lazy_run)
    # def run(self, *args, **kwargs):
    #     return super().run(*args, **kwargs)

    def _process_flat_group(self, flats: list[DetImage]) -> tuple[list[dict[str, Any]], list[DetImage]]:
        if len(flats) == 0:
            return [], []

        mask_key = self.meta.get("mask_key", "sigma_clip_mask")
        exptime = flats[0].image_type[self.exptime_key]
        det_id = flats[0].id
        stats = []
        for i, img in enumerate(flats):
            self.logger.info(f"Doing per output stats on #{i} image of {len(flats)}")
            for out_id, output in img.outputs.items():
                outstat = self.do_stats_per_output(output, mask_key=mask_key)
                outstat.update({"det_id": det_id, "output": out_id, "exptime": exptime, "diff": False})
                stats.append(outstat)

        #do differential processing on all diffs
        if len(flats) < 2:
            self.logger.warning(f"Only one image in the provided flat group for exptime {exptime} and det_id {det_id}, "
                                f"skipping diff analysis.")
            return [], []

        diff_images = []
        for diffpairidx in combinations(range(len(flats)), 2):
            dp1 = flats[diffpairidx[0]]
            dp2 = flats[diffpairidx[1]]
            self.logger.info("Doing diff pair analysis on index %d and %d", *diffpairidx)

            # init diff_img by copying one of the pair to transfer common attributes
            diff_img = deepcopy(dp1)
            diff_img.meta["diff_pair"] = f'{dp1.meta.get("filename", "unknown")},{dp2.meta.get("filename", "unknown")}'
            # Loop over each output, diff, and set in diff_img
            for out_id, output in diff_img.outputs.items():
                diffdat = output.data - dp2.outputs[out_id].data
                diffmask = xr.zeros_like(diffdat, dtype=bool)
                if self.meta.get("do_sigma_clip", True):
                    sigma_clip_args = {'sigma': 5.0, 'grow': 10, 'stdfunc': 'mad_std'}
                    sigma_clip_args.update(self.meta.get("sigma_clip_args", {}))
                    imslc = output.image_region
                    ma_diffdat = sigma_clip_image(slice_data(diffdat, imslc).values, **sigma_clip_args)
                    diffdat.loc[imslc] = ma_diffdat.filled(np.nan)
                    diffmask.loc[imslc] |= ma_diffdat.mask
                output.set_data_in_parent(diffdat)
                output.masks['sigma_clip_mask'] = diffmask
                diffstat = self.do_stats_per_output(output, mask_key=mask_key)
                diffstat.update({"det_id": det_id, "output": out_id, "exptime": exptime, "diff": True})
                stats.append(diffstat)
            diff_images.append(diff_img)

        return stats, diff_images

    def do_stats_per_output(self, output: CCDOutput, mask_key: str = 'sigma_clip_mask'):
        stats = {}

        imslc = output.image_region
        imarr = output.get_image_region()

        if hasattr(output, 'masks') and (mask_key in output.masks):
            mask = slice_data(output.masks[mask_key], imslc).values
            stats["n_masked"] = int(np.count_nonzero(mask))
        else:
            mask = np.zeros_like(imarr)
            stats["n_masked"] = 0
        ma_imarr = np.ma.masked_array(imarr.values, mask=mask)

        stats |= self.basic_stats(ma_imarr, "")
        stats |= self.stats_tests(ma_imarr, "")

        # parallel EPER slice, do not mask overscan
        llel_oscan = output.get_overscan(kind='parallel')
        stats |= self.calc_eper_trail(llel_oscan.values, axis=int(not output.parallel_axint), prepend_kw="llel_")

        # NOTE last row and col often masked by sigma clipping, so do these with normal (unmasked) stats.
        ## last => last one to readout => check against readout pixel
        ## renaming row/col to llel/ser
        last_inds = tuple(x-y-1 for x,y in zip(imarr.shape,output.readout_pixel))

        last_llel_slice = {output.parallel_axis: slice(last_inds[output.parallel_axint], last_inds[output.parallel_axint] + 1),
                          output.serial_axis: slice(None, None)}
        last_llel = slice_data(imarr, last_llel_slice)
        stats |= self.basic_stats(last_llel.values, "lastllel_")

        last_ser_slice = {output.parallel_axis: slice(None, None),
                          output.serial_axis: slice(last_inds[output.serial_axint], last_inds[output.serial_axint] + 1)}
        last_ser = slice_data(imarr, last_ser_slice)
        stats |= self.basic_stats(last_ser.values, "lastser_")

        # serial EPER slice and last column
        ser_oscan = output.get_overscan(kind='serial')
        stats |= self.calc_eper_trail(ser_oscan.values, axis=int(output.parallel_axint), prepend_kw="ser_")

        # overscan stats
        stats |= self.basic_stats(ser_oscan.values, "oscan_")
        stats |= self.stats_tests(ser_oscan.values, "oscan_")

        if self.PSD_size is not None:
            self.logger.info("Calculating correlation coefficients")
            correlfilled = ma_imarr.data
            PSD, dPSD = self.calculate_PSD(correlfilled)
            stats |= {"PSD": PSD, "dPSD": dPSD}

        return stats

    @staticmethod
    def basic_stats(data: np.ndarray | np.ma.MaskedArray, prepend_kw: str = "") -> dict[str, Any]:
        """
        Calculate basic statistics for the given data.
        :param data: np.ndarray or np.ma.MaskedArray
            Input data array.
        :param prepend_kw: str
            Prefix to prepend to the keys in the output dictionary.
        :return: dict[str, Any]
            Dictionary containing calculated statistics.
        """
        _operations: dict[str, Callable] = {"med": np.ma.median,
                                            "mean": np.ma.mean,
                                            "std": np.ma.std,
                                            "mad": partial(ma_mad, scale="normal")}

        out = {f"{prepend_kw}{kw}": float(op(data)) for kw, op in _operations.items()}
        return out

    @staticmethod
    def stats_tests(data: np.ndarray | np.ma.MaskedArray, prepend_kw: str = "") -> dict[str, Any]:
        """
        Calculate skewness and kurtosis for the given data.
        :param data: np.ndarray or np.ma.MaskedArray
            Input data array.
        :param prepend_kw: str
            Prefix to prepend to the keys in the output dictionary.
        :return: dict[str, Any]
        """

        _operations: dict[str, Callable] = {"skew" : ma_skew,
                                           "kurt" : ma_kurt}
        out = {f"{prepend_kw}{kw}" : float(op(data)) for kw, op in _operations.items()}

        tests: dict[str,Callable] = {"skewtest" : ma_skewtest,
                                     "kurttest": ma_kurttest}

        for teststr, testop in tests.items():
            stat, pval = testop(data)
            k = f"{prepend_kw}{teststr}"
            out[k] = float(stat)
            out[f"{k}p"] = float(pval)
        return out

    @staticmethod
    def calc_eper_trail(data: np.ndarray | np.ma.MaskedArray, axis: int, prepend_kw: str = "") -> dict[str, Any]:
        eper_med = np.median(data, axis=axis)
        eper_mean = np.mean(data, axis=axis)
        return {f"{prepend_kw}eper_med" : eper_med,
                f"{prepend_kw}eper_mean" : eper_mean}

    def calculate_PSD(self, imarr: np.ndarray):
        kwargs = self.meta.get('welch2d_kwargs', {})
        out = welch2d(imarr, self.PSD_size, **kwargs)
        return out