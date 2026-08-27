from copy import deepcopy
import pandas as pd
from typing import Generator, Any
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from itertools import combinations
from pydantic import field_validator
import os
import warnings
warnings.filterwarnings("ignore")

from eregion.utils import slice_data, save_dataframe_to_fits, load_dataframe_from_fits, decrease_slicer_stop_index
from eregion.datamodels import DetImage, Output, CCDOutput, ImageBundle, TaskResult
from eregion.tasks import LazyTask
from eregion.core.image_stats import do_statistics, STATFUNCS
from eregion.core.image_operations import sigma_clip_image
from eregion.core.welch2d import welch2d

##################### PTC task ############################
class PTCResult(TaskResult):
    ptc_table: pd.DataFrame
    diff_images: ImageBundle
    ptc_meta: dict[str, Any] = {}

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
        save_dataframe_to_fits(self.ptc_table, os.path.join(filepath, "ptc_table.fits"))
        self.params.update(self.ptc_meta)
        super().save(filepath)

    @classmethod
    def load(cls, filepath: str, load_diffs=False):
        if load_diffs:
            diff_images = ImageBundle.load(os.path.join(filepath, "diff_images"))
        else:
            diff_images = ImageBundle()
        ptc_table = load_dataframe_from_fits(os.path.join(filepath, "ptc_table.fits"))
        metadata = cls.load_metadata(filepath)
        return cls(ptc_table=ptc_table, diff_images=diff_images, ptc_meta=metadata.get("params", {}), **metadata)



class PTC(LazyTask):
    task_result = PTCResult

    def __init__(self,
                 psd_size: int = 9,
                 groupby_keys: list[str] = ['exptime', 'det_id'],
                 exptime_key: str = 'exptime',
                 name: str='PTC',
                 **kwargs
                 ):
        """
        Run photon transfer curve (PTC) analysis for input pairs of flat images.

        Parameters
        ----------
        psd_size: Optional[int]
            Size of power spectral density array, by default 9.
        groupby_keys: list[str]
            List of keys to group by when processing the images, by default ['exptime', 'det_id'].
        exptime_key: str
            Key for exposure time in the image metadata, by default 'exptime'.
        name: Optional[str]
            Name of the task. Default is 'PTC'.
        **kwargs: Optional[dict]
            Additional keyword arguments for PTC task, including:

            * mask_key (str): Key/data_var for the mask in the output CCDOutput.masks, by default 'sigma_clip_mask'. Used to mask pixels before calculating statistics on flats.
            * do_sigma_clip (bool): Whether to perform sigma clipping on differenced output data (only image region)
            * sigma_clip_args (dict[str, Any]): Additional arguments to pass to the sigma_clip function, e.g., {'sigma': 5.0, 'grow': 10}.
            * welch2d_kwargs (dict[str, Any]): Additional arguments to pass to the welch2d function,
        """
        super().__init__(name=name, **kwargs)
        self.PSD_size = psd_size
        self.groupby_keys = groupby_keys
        if 'det_id' not in self.groupby_keys:
            self.groupby_keys.append('det_id')
        self.exptime_key = exptime_key

    def lazy_run(self, images: ImageBundle | list[DetImage]) -> Generator[PTCResult, None, None]:
        """
        For each unique DetImage.id and exposure time, get flat pairs and derive ptc
        :param images: ImageBundle | list[DetImage]
            Input "flat" type images to process. Can be an ImageBundle or a list of DetImage objects.
        :return: Generator[PTCResult, None, None]
            A generator that yields PTCResult objects.
        """
        images = images if isinstance(images, ImageBundle) else ImageBundle(images)
        groups = images.groupby(by=self.groupby_keys, sort=False)

        results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._process_flat_group)(ImageBundle.from_dataframe(group)) for unique_key, group in groups
        )

        stats, diff_images = [], []
        for result in results:
            stats.extend(result[0])
            diff_images.extend(result[1])
        ptcdf = self.make_ptc_table(stats)

        self.logger.info(f"Processed {len(images)} flats, upto exposure time {max(images.list[self.exptime_key])}")
        meta = self.meta | {"psd_size": self.PSD_size, "groupby_keys": self.groupby_keys, "exptime_key": self.exptime_key}
        yield self.task_result(ptc_table=ptcdf, diff_images=diff_images, ptc_meta=meta)

    def _process_flat_group(self, flats: ImageBundle[DetImage]) -> tuple[list[dict[str, Any]], list[DetImage]]:
        if len(flats) == 0:
            return [], []

        mask_key = self.meta.get("mask_key", "sigma_clip_mask")
        unique_info = {key: flats.list[key][0] for key in self.groupby_keys}
        exptime = unique_info[self.exptime_key]
        det_id = unique_info['det_id']
        self.logger.info(f"Processing {len(flats)} flats with exptime {exptime} and det_id {det_id}")

        stats = []
        for i, img in enumerate(flats):
            self.logger.debug(f"Doing per output stats on image #{i} of {len(flats)}")
            for out_id, output in img.outputs.items():
                outstat = (unique_info | # Add unique keys identifying flat group
                           {"output": out_id, "diff": False, "seqnum":str(i)} | # Add output id, # img index, and diff flag
                           self.do_stats_per_output(output, mask_key=mask_key))  # Add statistics for this output
                stats.append(outstat)

        # do differential processing on all diffs
        if len(flats) < 2:
            self.logger.warning(f"Only one image in the provided flat group with exptime {exptime} and det_id {det_id}, "
                                f"skipping diff analysis.")
            return stats, []

        diff_images = []
        for diffpairidx in combinations(range(len(flats)), 2):
            dp1 = flats[diffpairidx[0]]
            dp2 = flats[diffpairidx[1]]
            self.logger.debug("Doing diff pair analysis on index %d and %d", *diffpairidx)

            # create a new DetImage for the diff pair
            diff_img = deepcopy(dp1)
            diff_img.meta["diff_pair"] = f'{dp1.meta.get("filename", "unknown")},{dp2.meta.get("filename", "unknown")}'

            # Loop over each output, diff, and set in diff_img
            for out_id, output in diff_img.outputs.items():
                diffdat = output.data - dp2.outputs[out_id].data # diff of two outputs
                output.set_data_in_parent(diffdat) # set in parent

                diffmask = xr.zeros_like(diffdat, dtype=bool) # initialize diff mask
                if self.meta.get("do_sigma_clip", True):
                    sigma_clip_args = {'sigma': 5.0, 'grow': 10, 'stdfunc': 'mad_std'}
                    sigma_clip_args.update(self.meta.get("sigma_clip_args", {}))
                    # sigma clip only image region
                    imslc = output.image_region
                    ma_diffdat = sigma_clip_image(slice_data(diffdat, imslc).values, **sigma_clip_args)
                    output.set_data_in_parent(ma_diffdat.filled(np.nan), imslc) # set nan filled data in parent
                    diffmask.loc[decrease_slicer_stop_index(imslc)] |= ma_diffdat.mask # set in diff mask
                output.masks = diffmask.to_dataset(name='sigma_clip_mask')  # set in output attr

                diffstat = (unique_info | # Add unique keys identifying flat group
                           {"output": out_id, "diff": True, "seqnum":'-'.join([str(i) for i in diffpairidx])} |
                           self.do_stats_per_output(output, mask_key='sigma_clip_mask'))  # Add statistics for this output
                stats.append(diffstat)
            diff_images.append(diff_img)

        return stats, diff_images

    def do_stats_per_output(self, output: Output, mask_key: str = 'sigma_clip_mask'):
        stats = {}

        imarr, immask = output.get_image_region(return_masks=True)

        # basic and extra stats on masked image region, should work for all type of outputs
        if immask is not None and mask_key in immask:
            mask = immask[mask_key].values
            stats["n_masked"] = int(np.count_nonzero(mask))
        else:
            mask = np.zeros_like(imarr)
            stats["n_masked"] = 0
        ma_imarr = np.ma.masked_array(imarr.values, mask=mask)
        stats |= do_statistics(ma_imarr, which=STATFUNCS, axis=None, prepend_kw="")

        # power spectral density (correlation coefficients) on masked image region
        if self.PSD_size is not None:
            self.logger.info("Calculating correlation coefficients")
            correlfilled = ma_imarr.data
            PSD, dPSD = self.calculate_PSD(correlfilled)
            stats |= {"PSD": PSD, "dPSD": dPSD}

        ###### EXTRA stats, CCD specific ########
        if isinstance(output, CCDOutput):
            llel_oscan = output.get_overscan(kind='parallel').values
            ser_oscan = output.get_overscan(kind='serial').values

            # overscan stats,
            stats |= do_statistics(data=ser_oscan, which=STATFUNCS, axis=None, prepend_kw="ser_oscan_")
            stats |= do_statistics(data=llel_oscan, which=STATFUNCS, axis=None, prepend_kw="llel_oscan_")

            # parallel and serial EPER slice, do not mask overscan
            EPERSTATS = {'eper_median': STATFUNCS['median'], 'eper_mean': STATFUNCS['mean']}
            stats |= do_statistics(data=llel_oscan, which=EPERSTATS, axis=int(not output.parallel_axint), prepend_kw="llel_")
            stats |= do_statistics(data=ser_oscan, which=EPERSTATS, axis=int(not output.serial_axint), prepend_kw="ser_")

            BASICSTATS = {'mean': STATFUNCS['mean'], 'median': STATFUNCS['median'], 'std': STATFUNCS['std'],
                          'mad': STATFUNCS['mad']}
            # NOTE last row and col often masked by sigma clipping, so do these with normal (unmasked) stats.
            ## last => last one to readout => check against readout pixel
            ## renaming row/col to llel/ser
            last_llel_arr = imarr.sel({output.parallel_axis: output.parallel_overscan.start - output.parallel_overscan.step})
            stats |= do_statistics(data=last_llel_arr.values, which=BASICSTATS, axis=None, prepend_kw="lastllel_")

            last_ser_arr = imarr.sel({output.serial_axis: output.serial_overscan.start - output.serial_overscan.step})
            stats |= do_statistics(data=last_ser_arr.values, which=BASICSTATS, axis=None, prepend_kw="lastser_")

        return stats

    def calculate_PSD(self, imarr: np.ndarray):
        kwargs = self.meta.get('welch2d_kwargs', {})
        out = welch2d(imarr, self.PSD_size, **kwargs)
        return out

    def make_ptc_table(self, stats: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Create a flattened PTC table from the provided statistics.
        :param stats: list[dict[str, Any]]
            List of dictionaries containing statistics for each output.
        :return: pd.DataFrame
            DataFrame containing the PTC table.
        """
        df = pd.DataFrame(stats)

        fixed_cols = list(self.groupby_keys) + ['output', 'diff', 'seqnum']
        df['seqnum'] = df["seqnum"].astype(str)

        ptc_tab = []
        groupby_keys = self.groupby_keys + ['output']
        for unique_keys, group in df.groupby(by=groupby_keys, sort=False):
            gdf = group.reset_index(drop=True)
            row = {key: gdf[key][0] for key in groupby_keys}
            for i in range(len(gdf)):
                row.update({f"{col}_{gdf['seqnum'][i]}": gdf.iloc[i][col] for col in df.columns if col not in fixed_cols})
            ptc_tab.append(row)
        ptc_df = pd.DataFrame(ptc_tab)

        file_sfx = df[~df['diff']]['seqnum'].unique()
        diff_sfx = df[df['diff']]['seqnum'].unique()
        ptc_df['mean'] = ptc_df[[f'mean_{sfx}' for sfx in file_sfx]].mean(axis=1)
        ptc_df['median'] = ptc_df[[f'median_{sfx}' for sfx in file_sfx]].mean(axis=1)
        ptc_df['std'] = ptc_df[[f'std_{sfx}' for sfx in diff_sfx]].mean(axis=1)/np.sqrt(2) # divide by sqrt(2) because this noise is from difference of two images
        return ptc_df