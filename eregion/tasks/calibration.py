from copy import deepcopy
import numpy as np
import os
import json
from pydantic import Field, model_validator, ConfigDict

from tasks import Task
from datamodels import DetImage, ImageBundle, TaskResult

# Dataclass to hold master bias results
class CalibrationResult(TaskResult):
    master_bias: ImageBundle = Field(default=ImageBundle(),
                                     description="Bundle of master bias frames generated from input bias images.")
    master_dark: ImageBundle = Field(default=ImageBundle(),
                                     description="Bundle of master dark frames generated from input dark images.")
    master_flat: ImageBundle = Field(default=ImageBundle(),
                                     description="Bundle of master flat frames generated from input flat images.")
    master_lamp: ImageBundle = Field(default=ImageBundle(),
                                     description="Bundle of master lamp frames generated from input lamp images.")
    other: ImageBundle = Field(default=ImageBundle(),
                               description="Bundle of other calibration frames generated from input cal images.")
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode='before')
    @classmethod
    def parse_result(cls, kwargs):
        payload_fields = cls.payload_field_names()
        for key, val in kwargs.items():
            if key not in payload_fields:
                continue
            if not isinstance(val, ImageBundle):
                kwargs[key] = ImageBundle(val)
        return kwargs

    def save(self, filepath: str, **kwargs) -> None:
        for attr, value in self.payload_dict().items():
            if isinstance(value, ImageBundle):
                value.save(os.path.join(filepath, f"{attr}"), **kwargs)
        super().save(filepath)

    @classmethod
    def load(cls, filepath: str):
        attrs = {}
        for attr in cls.payload_field_names():
            attrs[attr] = ImageBundle.load(os.path.join(filepath, f"{attr}"))
        with open(os.path.join(filepath, f"{cls.__name__}_metadata.json"), "r") as f:
            metadata = json.load(f)
        return cls(**attrs, **metadata)

# Task to generate master bias
class MasterBias(Task):
    task_result = CalibrationResult

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def run(self, bias_images: ImageBundle | list, **kwargs) -> CalibrationResult:
        """
        Generate master bias frames from a list of bias DetImage objects.
        :param bias_images: list of DetImage
            List of detector images containing bias frames.
        :return: {"master_biases": list[DetImage]}
            Dictionary with key 'master_biases' containing a list of master bias DetImage objects.
        """
        # Check that bias_images are DetImage instances
        bias_images = bias_images if isinstance(bias_images, ImageBundle) else ImageBundle(bias_images)

        # Get available detectors
        unique_dets = bias_images.list['det_id'].unique()

        # Create master bias for each detector
        master_biases = []
        for det_id in unique_dets:
            imgs = bias_images.filter(f'det_id == "{det_id}"')
            # Initialize master bias DetImage
            master_bias = deepcopy(imgs[0])
            master_bias.meta['filenames'] = ', '.join([img.meta['filename'] for img in imgs])
            master_bias.image_type.update({'type': 'master_bias'})
            # Combine bias data
            biases = [img.data.values for img in imgs]
            mb = self._create_masterbias(biases, method=kwargs.get('method', 'median'))
            master_bias.set_data(mb)
            master_biases.append(master_bias)

        return self.task_result(master_bias=ImageBundle(master_biases))

    def _create_masterbias(self, biases: list[np.ndarray], method='median')-> np.ndarray:
        """
        Create a master bias frame from a list of bias frames using the specified method.
        :param biases: list of numpy arrays
            List of detector images containing bias frames.
        :param method: str
            Method to combine bias frames. Currently only 'median' is implemented.
        :return: master_bias: numpy array
            The generated master bias frame.
        """
        if self.method_name != method:
            self.set_method(method)
        return self.method(biases)

    @property
    def methods(self):
        """
        Return a dictionary of available methods for creating master bias and their function signatures.
        :return: dict
            Dictionary with method names as keys and function signatures as values.
        """
        return {
            'median': 'core.image_operations.median_combine',
        }

    def __call__(self, biases: list[np.ndarray],  method='median') -> np.ndarray:
        return self._create_masterbias(biases, method=method)

