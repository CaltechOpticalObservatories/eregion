import numpy as np
import xarray as xr
from typing import Union
from numpy.typing import NDArray

from tasks.task import Task
from datamodels.image import DetImage, Output
from datamodels.image_utils import ensure_dataarray
from core.image_operations import median_combine


# Task to generate master bias
class MasterBias(Task):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def run(self, bias_images: list[DetImage]) -> DetImage:
        """
        Generate a master bias frame from a list of bias DetImage objects.
        :param bias_images: list of DetImage
            List of detector images containing bias frames.
        :return: master_bias: DetImage
            The generated master bias frame.
        """
        # Check that bias_images are DetImage instances
        if not isinstance(bias_images, list) or not all(isinstance(img, DetImage) for img in bias_images):
            raise ValueError("bias_images must be a list of DetImage instances.")

        # Initialize master bias DetImage
        master_bias = DetImage(image_type="master_bias")
        # Combine bias data
        bias_data = [bias_image.data.values for bias_image in bias_images]
        master_bias.data = ensure_dataarray(self._create_masterbias(bias_data,
                                                                    method=self.meta.get('method', 'median')))
        # Copy metadata and outputs from the first bias image
        master_bias.meta = bias_images[0].meta
        master_bias.focal_plane = bias_images[0].focal_plane
        master_bias.meta["filenames"] = ', '.join([img.meta.get("filename", "unknown") for img in bias_images])
        master_bias.outputs.update(bias_images[0].outputs)
        return master_bias

    def _create_masterbias(self, biases: list[NDArray], method='median')-> NDArray:
        """
        Create a master bias frame from a list of bias frames using the specified method.
        :param biases: list of numpy arrays
            List of detector images containing bias frames.
        :param method: str
            Method to combine bias frames. Currently only 'median' is implemented.
        :return: master_bias: numpy array
            The generated master bias frame.
        """
        if method == 'median':
            return median_combine(biases)
        else:
            # print available methods
            self.print_methods()
            raise NotImplementedError

    @property
    def methods(self):
        """
        Return a dictionary of available methods for creating master bias and their function signatures.
        :return: dict
            Dictionary with method names as keys and function signatures as values.
        """
        return {
            'median': 'core.image_operations.median_combine(images: list[np.ndarray]) -> np.ndarray',
        }


    def __call__(self, biases: list[NDArray],  method='median') -> NDArray:
        return self._create_masterbias(biases, method=method)

