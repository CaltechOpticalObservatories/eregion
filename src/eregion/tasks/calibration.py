import pandas as pd
from copy import deepcopy
import numpy as np
from pydantic import Field, ConfigDict

from eregion.tasks import Task
from eregion.datamodels import DetImage, ImageBundle, FPImageBundle
from eregion.tasks.imagegen import ImageResult
from eregion.core.image_stats import do_statistics, STATFUNCS

# Dataclass to hold master bias results
class CalibrationResult(ImageResult):
    """
    A dataclass to hold the results of a calibration task.

    Attributes:
        master_bias (ImageBundle | FPImageBundle): Bundle of master bias frames generated from MasterBias task.
        master_dark (ImageBundle | FPImageBundle): Bundle of master dark frames generated from MasterDark task.
        master_flat (ImageBundle | FPImageBundle): Bundle of master flat frames generated from MasterFlat task.
        master_lamp (ImageBundle | FPImageBundle): Bundle of master lamp frames generated from MasterLamp task.
        data (ImageBundle | FPImageBundle): Bundle of other calibration frames generated from misc tasks.
        stats (pd.DataFrame): DataFrame containing statistics of the calibration frames if calculated in the task.
    """
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
    required_keys = ['groupby_keys']

    def __init__(self, name=None, **kwargs):
        """
        Initialize the MasterCombine task with optional groupby keys for combining calibration frames.
        :param name: str
            Name of the task.
        :param kwargs: dict, optional
        :keyword groupby_keys: list
            List of keys to group by when combining calibration frames. Defaults to ['det_id', 'type'] if not provided.
        """
        if kwargs.get('groupby_keys', None) is None:
            kwargs['groupby_keys'] = ['det_id', 'type']
        else:
            groupby = kwargs['groupby_keys'] if isinstance(kwargs['groupby_keys'], list) else [kwargs['groupby_keys']]
            groupby = groupby + ['det_id'] # in case people forget that combining by det_id is obvious here
            kwargs['groupby_keys'] = list(dict.fromkeys(groupby)) # remove duplicates while preserving order
        super().__init__(name=name, **kwargs)

    def run(self, images: ImageBundle | list, **kwargs) -> ImageResult:
        """
        Generate master calibration frames from a list of DetImage objects.
        :param images: list of DetImage
            List of detector images containing calibration frames (bias, dark, flat, lamp) to combine.
        :return: ImageResult
            An ImageResult object containing the generated master calibration frames.
        """
        # Validate images
        images = images if isinstance(images, ImageBundle) else ImageBundle(images)

        # Make groups to combine
        groupkeys = self.meta['groupby_keys']
        groups = images.groupby(by=groupkeys)

        # Create master calibration for each group
        master_cals = []
        for _, group in groups:
            unique_keys = [str(group.iloc[0][key]) for key in groupkeys if key != 'det_id']
            unique_id = '_'.join(unique_keys)

            imgs = list(group['object'])
            # Initialize master calibration DetImage
            master_cal = deepcopy(imgs[0])
            master_cal.meta['filename'] = ', '.join([img.meta['filename'] for img in imgs])
            master_cal.image_type.update({'type': f'master_{unique_id}'})
            # Combine calibration data
            cal_data = [img.data.values for img in imgs]
            mc = self._create_mastercal(cal_data, method=kwargs.get('method', 'median'))
            master_cal.set_data(mc)
            master_cals.append(master_cal)
            self.logger.info(f'Created master calibration frame for det_id: {master_cal.id}, type: {master_cal.image_type}')

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
    """
    Task to generate a master bias frame from a list of bias frames.
    """
    task_result = CalibrationResult

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def run(self, images: ImageBundle | list, add_to: CalibrationResult = None, **kwargs) -> CalibrationResult:
        """
        Generate a master bias frame from a list of bias frames and optionally combine it with an existing CalibrationResult.
        :param images: list/bundle of images to combine
        :param add_to: CalibrationResult, optional
            An existing CalibrationResult to combine the new master bias with. If None, a new CalibrationResult is created.
        :param kwargs: Additional keyword arguments to pass to the parent run method.
        :return: CalibrationResult
        """
        res = super().run(images, **kwargs)
        calres = self.task_result(master_bias=res.data)
        calres = add_to.combine(calres) if add_to is not None else calres
        return calres

# Task to generate master dark
class MasterDark(MasterCombine):
    """
    Task to generate a master dark frame from a list of dark frames.
    """
    task_result = CalibrationResult

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def run(self, images: ImageBundle | list, add_to: CalibrationResult = None, mask_key='sigma_clip_mask', **kwargs) -> CalibrationResult:
        """
        Generate a master dark frame from a list of dark frames, calculate statistics for each output, and optionally combine it with an existing CalibrationResult.
        :param images: list/bundle of images to combine
        :param add_to: CalibrationResult, optional
            An existing CalibrationResult to combine the new master dark with. If None, a new CalibrationResult is created.
        :param mask_key: str, optional
            The key to use for the mask in the .masks attribute.
        :param kwargs: Additional keyword arguments to pass to the parent run method.
        :return: CalibrationResult
        """
        res = super().run(images, **kwargs)
        darks = res.data

        stats = []
        for i, img in enumerate(darks):
            self.logger.info(f"Doing per output stats on image #{i} of {len(darks)}")
            imtype = {key: darks.list.iloc[i][key] for key in self.meta['groupby_keys']}
            for out_id, output in img.outputs.items():
                outstat = imtype | self.do_stats_per_output(output, mask_key=mask_key)
                stats.append(outstat)
        stats = pd.DataFrame(stats)

        calres = self.task_result(master_dark=darks, stats=stats)
        calres = add_to.combine(calres) if add_to is not None else calres
        return calres

    def do_stats_per_output(self, output, mask_key='sigma_clip_mask'):
        stats = {"output": output.id}
        imarr, immask = output.get_image_region(return_masks=True)
        if immask is not None and mask_key in immask:
            mask = immask[mask_key].values
            stats["n_masked"] = int(np.count_nonzero(mask))
        else:
            mask = np.zeros_like(imarr)
            stats["n_masked"] = 0
        ma_imarr = np.ma.masked_array(imarr.values, mask=mask)
        stats |= do_statistics(ma_imarr, which=STATFUNCS, prepend_kw="")
        return stats
