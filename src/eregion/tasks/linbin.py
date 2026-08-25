"""
Linearity analysis following the method in:
    Stephen Kaye, Roger Smith, Peter H. Mao, et al.
    "CCD linearity measurement by incremental binning (Conference Presentation)", Proc. SPIE 10709,
    High Energy, Optical, and Infrared Detectors for Astronomy VIII, 107090Z (10 Jul 2018);
    https://doi.org/10.1117/12.2314251

"""
from eregion.tasks import LazyTask
from eregion.datamodels import TaskResult, ImageBundle, DetImage, CCDOutput
from eregion.core.image_operations import do_digital_binning
from eregion.core.image_stats import do_statistics, STATFUNCS
from eregion.utils import slice_data, decrease_slicer_stop_index

from pydantic import Field
from joblib import Parallel, delayed
from typing import Callable, Generator
import numpy as np
import pandas as pd

class LinBinResult(TaskResult):
    stats: pd.DataFrame = Field(default_factory=pd.DataFrame)


class LinBin(LazyTask):
    """
    Linearity measurement by incremental binning (https://doi.org/10.1117/12.2314251). Two sets of flat frames are
    required: one standard, and one readout with incrementally binned rows in parallel direction.
    The standard flats are digitally binned to match the binning of the analog-binned flats
    and compared to determine linearity of the CCD.

    The binning is assumed to start at the image region in parallel direction, i.e. after parallel prescan.

    The result of this task is a dataframe with stats per output channel per image.
    """
    task_result = LinBinResult

    def __init__(self,
                 binsizes: int | Callable[[int | None], int] = 1,
                 groupby_keys: list[str] = ["det_id"],
                 name: str = "LinBin", **kwargs):
        """
        Initialize the LinBin task.
        :param binsizes: int | Callable,
            If integer, increment the number of rows to sum per bin by it. If Callable, a function that yields the next binning value.
        :param groupby_keys: list[str]
            The keys to group the images by to identify pairs of related standard and linbin flats.
        :param name: str, optional
            The name of the task. Default is "LinBin".
        :param kwargs: Additional keyword arguments.

        """
        super().__init__(name=name, **kwargs)

        if isinstance(binsizes, int):
            binsizes = lambda previous: previous + binsizes if previous is not None else binsizes
        elif isinstance(binsizes, Callable):
            pass  # Use the provided callable directly
        else:
            raise ValueError("binsizes must be an integer or a callable function that returns integer.")
        self.binsizes = binsizes
        self.bins = None
        self.groupby_keys = groupby_keys

    def lazy_run(self,
                 normal_flats: ImageBundle | list[DetImage],
                 linbin_flats: ImageBundle | list[DetImage],
                 **kwargs) -> Generator[LinBinResult, None, None]:
        """
        Run the LinBin task on the provided images.
        :param normal_flats: ImageBundle | list[DetImage],
            The standard flat frames to be digitally binned.
        :param linbin_flats: ImageBundle | list[DetImage],
            The flat frames with incrementally binned rows in parallel direction.
        :return: yields LinBinResult,
            The result of the LinBin task containing the statistics dataframe.
        """
        normal_flats = normal_flats if isinstance(normal_flats, ImageBundle) else ImageBundle(normal_flats)
        linbin_flats = linbin_flats if isinstance(linbin_flats, ImageBundle) else ImageBundle(linbin_flats)

        # digital binning of the normal flats to match the linbin flats
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._bin_image)(normal_flat) for normal_flat in normal_flats
        )
        digbin_flats = ImageBundle(results)


        pairstats = []
        for unique_keys, dig_group in digbin_flats.groupby(by=self.groupby_keys):
            if len(dig_group) == 0:
                self.logger.warning(f"No normal flats found for {self.groupby_keys} == {unique_keys}. Skipping.")
                continue

            query = ' '.join([f'{key} == "{val}"' for key,val in zip(self.groupby_keys, unique_keys)])
            lin_group = linbin_flats(query).list # get the linbin flats matching the same group (.list returns the DataFrame)
            if len(lin_group) == 0:
                self.logger.warning(f"No linbin flats found for {self.groupby_keys} == {unique_keys}. Skipping.")
                continue

            # stats per output channel for the digbin and linbin flats
            stats = {key: dig_group.iloc[0][key] for key in self.groupby_keys}
            for i, img in enumerate(dig_group['object']):  # dig_group is a DataFrame
                outstat = stats.copy()
                for out_id, output in img.outputs.items():
                    outstat |= {"seqnum": str(i), "binning": "digital"} | self.do_stats_per_output(output)
                pairstats.append(outstat)

            for i, img in enumerate(lin_group['object']):  # lin_group is a DataFrame
                outstat = stats.copy()
                for out_id, output in img.outputs.items():
                    outstat |= {"seqnum": str(i), "binning": "analog"} | self.do_stats_per_output(output)
                pairstats.append(outstat)

        stats = self.make_linbin_table(pairstats)
        yield self.task_result(stats=stats)

    def _bin_image(self, img: DetImage):
        if not all(isinstance(output, CCDOutput) for output in img.outputs.values()):
            raise ValueError("Outputs must be of type CCDOutput")

        mask_key = self.meta.get("mask_key", "sigma_clip_mask")

        for output in img.outputs.values():
            # Ensure we take all serial pixels and not parallel prescan/overscan
            imslc = output.image_region
            imslc[output.serial_axis] = slice(None)
            # Slice the data and mask for the current output, slice direction is from prescan to overscan, so that the first row is the first row read out from the CCD
            imdata = slice_data(output.data, imslc).values
            immask = slice_data(output.masks[mask_key], imslc).values if (output.masks is not None and mask_key in output.masks) else None
            # get bins
            self._get_binsizes(imdata.shape[output.parallel_axint])
            # bin data
            binned_data = do_digital_binning(imdata, binsizes=self.bins, binaxis=output.parallel_axint)
            output.set_data_in_parent(binned_data, imslc)
            # bin mask if it exists
            if immask is not None:
                binned_mask = do_digital_binning(immask.astype(int), binsizes=self.bins, binaxis=output.parallel_axint).astype(bool)
                _imslc = decrease_slicer_stop_index(imslc)
                output.masks[mask_key].loc[_imslc] = binned_mask

        return img

    def _get_binsizes(self, parallel_size: int):
        if self.bins is None:
            bins = []
            while np.sum(bins) < parallel_size:
                next_bin = self.binsizes(bins[-1]) if len(bins) > 0 else self.binsizes(None)
                bins.append(int(next_bin))
            if np.sum(bins) > parallel_size:
                self.logger.warning(f"Sum of bins {np.sum(bins)} exceeds parallel size {parallel_size}. Adjusting last bin size.")
                bins[-1] -= (np.sum(bins) - parallel_size)
            self.bins = bins

    def do_stats_per_output(self, output: CCDOutput, mask_key: str = 'sigma_clip_mask'):
        imarr, immask = output.get_image_region(return_masks=True)
        stats = {'output':output.id, 'bins': self.bins}
        if immask is not None and mask_key in immask:
            mask = immask[mask_key].values
            stats["n_masked"] = int(np.count_nonzero(mask))
        else:
            mask = np.zeros_like(imarr)
            stats["n_masked"] = 0
        ma_imarr = np.ma.masked_array(imarr.values, mask=mask)[0:len(self.bins)] # only take the binned rows, i.e. the first len(bins) rows

        BASICFUNCS = {'mean': STATFUNCS['mean'], 'median': STATFUNCS['median'],
                      'std': STATFUNCS['std'], 'mad': STATFUNCS['mad']}
        stats |= do_statistics(ma_imarr, which=BASICFUNCS, axis=output.serial_axint, prepend_kw='')

        # serial overscan region stats, unmasked
        imslc = output.image_region[output.parallel_axis]
        ser_oscan = slice_data(output.get_overscan(kind='serial'), {'y': imslc}).values[0:len(self.bins)]
        stats |= do_statistics(ser_oscan, which=BASICFUNCS, axis=output.serial_axint, prepend_kw='ser_oscan_')

        return stats

    def make_linbin_table(self, stats) -> pd.DataFrame:
        stats = pd.DataFrame(stats)
        groupkeys = self.groupby_keys + ['output', 'bins']
        stats['suffix'] = stats['binning'] + '_' + stats['seqnum']
        stats = stats.drop(columns=['seqnum', 'binning'])
        statcols = [col for col in stats.columns if col not in groupkeys + ['suffix']]

        grouped = stats.groupby(groupkeys)
        linbin_table = []
        for _, group in grouped:
            row = {key: group.iloc[0][key] for key in groupkeys}
            for i in range(len(group)):
                row |= {f"{k}_{group.iloc[i]['suffix']}": group.iloc[i][k] for k in group.columns if k in statcols}
            linbin_table.append(row)
        return pd.DataFrame(linbin_table)




