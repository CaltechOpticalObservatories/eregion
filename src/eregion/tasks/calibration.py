import pandas as pd
from copy import deepcopy
import numpy as np
import os
from pydantic import Field, ConfigDict

from eregion.tasks import Task
from eregion.datamodels import DetImage, ImageBundle, FPImageBundle
from eregion.tasks.imagegen import ImageResult
from eregion.core.image_stats import basic_stats, stats_tests
from eregion.utils import slice_data

# Dataclass to hold master bias results
class CalibrationResult(ImageResult):
    """A dataclass to hold the results of a calibration task."""
    master_bias: ImageBundle | FPImageBundle = Field(default=ImageBundle(),
                                     description="Bundle of master bias frames generated from input bias images.")
    master_dark: ImageBundle | FPImageBundle = Field(default=ImageBundle(),
                                     description="Bundle of master dark frames generated from input dark images.")
    master_flat: ImageBundle | FPImageBundle = Field(default=ImageBundle(),
                                     description="Bundle of master flat frames generated from input flat images.")
    master_lamp: ImageBundle | FPImageBundle = Field(default=ImageBundle(),
                                     description="Bundle of master lamp frames generated from input lamp images.")
    data: ImageBundle | FPImageBundle = Field(default=ImageBundle(),
                               description="Bundle of other calibration frames generated from input cal images.")
    stats: pd.DataFrame = Field(default=pd.DataFrame(),
                                description="DataFrame containing statistics of the calibration frames.")
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

######## Master combine task -- methodology is same for bias, dark, flat so should be one task ###############
class MasterCombine(Task):
    task_result = ImageResult
    required_keys = ['group_by']

    def __init__(self, name=None, **kwargs):
        if kwargs.get('group_by', None) is None:
            kwargs['group_by'] = ['det_id', 'type']
        else:
            groupby = kwargs['group_by'] if isinstance(kwargs['group_by'], list) else [kwargs['group_by']]
            groupby = groupby + ['det_id'] # in case people forget that combining by det_id is obvious here
            kwargs['group_by'] = list(dict.fromkeys(groupby)) # remove duplicates while preserving order
        super().__init__(name=name, **kwargs)

    def run(self, images: ImageBundle | list, **kwargs) -> ImageResult:
        """
        Generate master calibration frames from a list of DetImage objects.
        :param images: list of DetImage
            List of detector images containing calibration frames (bias, dark, flat, lamp) to combine.
        :return: populated CalibrationResult
        """
        # Validate images
        images = images if isinstance(images, ImageBundle) else ImageBundle(images)

        # Make groups to combine
        groupkeys = self.meta['group_by']
        for key in groupkeys:
            if key not in images.list.columns:
                self.logger.error(f'Passed grouping key {key} is not present in image_type of the images, choose from'
                                  f' {images.list.columns}')
                raise KeyError(f'Passed grouping key {key} is not present in ImageBundle table {images.list.columns}')

        groups = images.list.groupby(groupkeys)
        det_id_index = np.where(np.array(groupkeys)=='det_id')[0][0]

        # Create master calibration for each group
        master_cals = []
        for group in groups:
            unique_id = list(group[0])
            unique_id.pop(det_id_index)
            unique_id = '_'.join(unique_id)

            imgs = list(group[1]['object'])

            # Initialize master calibration DetImage
            master_cal = deepcopy(imgs[0])
            master_cal.meta['filename'] = ', '.join([img.meta['filename'] for img in imgs])
            master_cal.image_type.update({'type': f'master_{unique_id}'})
            # Combine calibration data
            cal_data = [img.data.values for img in imgs]
            mc = self._create_mastercal(cal_data, method=kwargs.get('method', 'median'))
            master_cal.set_data(mc)
            master_cals.append(master_cal)
            self.logger.info(f'Created master calibration frame for {group[0]}')

        return self.task_result(data=master_cals)

    def _create_mastercal(self, images: list[np.ndarray], method='median')-> np.ndarray:
        """
        Create a master calibration frame from a list of calibration frames using the specified method.
        :param images: list of numpy arrays
            List of detector images containing calibration frames.
        :param method: str
            Method to combine calibration frames. Currently only 'median' is implemented.
        :return: master_cal: numpy array
            The generated master calibration frame.
        """
        if self.method_name != method:
            self.set_method(method)
        return self.method(images)

    @property
    def methods(self):
        """
        Return a dictionary of available methods for creating master cal and their function signatures.
        :return: dict
            Dictionary with method names as keys and function signatures as values.
        """
        return {
            'median': 'core.image_operations.median_combine',
        }

    def __call__(self, images: list[np.ndarray],  method='median') -> np.ndarray:
        return self._create_mastercal(images, method=method)

# Task to generate master bias
class MasterBias(MasterCombine):
    task_result = CalibrationResult

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def run(self, images: ImageBundle | list, add_to: CalibrationResult = None, **kwargs) -> CalibrationResult:
        res = super().run(images, **kwargs)
        calres = CalibrationResult(master_bias=res.data)
        if add_to is not None:
            calres = add_to.combine(calres)
        return calres

# Task to generate master dark
class MasterDark(MasterCombine):
    task_result = CalibrationResult

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def run(self, images: ImageBundle | list, add_to: CalibrationResult = None, mask_key='sigma_clip_mask', **kwargs) -> CalibrationResult:
        res = super().run(images, **kwargs)
        stats = []
        imtype_df = res.data.list
        for i, img in enumerate(res.data):
            self.logger.info(f"Doing per output stats on #{i} image of {len(res.data)}")
            imtype = imtype_df.iloc[i].to_dict()
            imtype.pop('object')
            for out_id, output in img.outputs.items():
                outstat = self.do_stats_per_output(output, mask_key=mask_key)
                outstat.update(imtype)
                stats.append(outstat)
        stats = pd.DataFrame(stats)

        calres = CalibrationResult(master_dark=res.data, stats=stats)
        if add_to is not None:
            calres = add_to.combine(calres)
        return calres

    def do_stats_per_output(self, output, mask_key='sigma_clip_mask'):
        stats = {"output": output.id}
        imslc = output.image_region
        imarr = output.get_image_region()
        if hasattr(output, 'masks') and (mask_key in output.masks):
            mask = slice_data(output.masks[mask_key], imslc).values
            stats["n_masked"] = int(np.count_nonzero(mask))
        else:
            mask = np.zeros_like(imarr)
            stats["n_masked"] = 0
        ma_imarr = np.ma.masked_array(imarr.values, mask=mask)
        stats |= basic_stats(ma_imarr, "")
        stats |= stats_tests(ma_imarr, "")
        return stats
