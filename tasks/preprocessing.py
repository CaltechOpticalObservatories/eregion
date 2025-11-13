import copy
from tasks.task import Task
from datamodels.image import DetImage, Output
import numpy as np


class BiasSubtraction(Task):
    required_keys = ["master_bias"]
    depends_on = ["MasterBiasTask"]

    def __init__(self, name=None, **kwargs):
        self.master_bias = kwargs["master_bias"]
        kwargs.pop("master_bias")
        super().__init__(name=name, **kwargs)

    def run(self, image: DetImage) -> DetImage:
        """
        Subtract the master bias from the given image.
        :param image: DetImage
            The science image from which to subtract the bias.
        :return: DetImage
            The bias-subtracted science image.
        """
        if isinstance(image, DetImage):
            # initialize a new DetImage for the bias-subtracted result as a copy of the input image
            bias_subtracted = copy.deepcopy(image)
            bias_subtracted.data = self._subtract_bias(image.data)
            bias_subtracted.meta["bias_subtracted"] = True
            bias_subtracted.image_type = f'bias_sub_{image.image_type}'
            return bias_subtracted
        else:
            raise ValueError("image must be an instance of DetImage, for numpy arrays use the __call__ method.")

    def _subtract_bias(self, image_data: np.ndarray) -> np.ndarray:
        """
        Algorithm that handles the bias subtraction.
        :param image_data: np.ndarray
        :return: np.ndarray
        """
        if isinstance(self.master_bias, DetImage):
            master_bias_data = self.master_bias.data
        elif isinstance(self.master_bias, np.ndarray):
            master_bias_data = self.master_bias
        else:
            raise ValueError("master_bias must be either a DetImage or a numpy array.")

        # Subtract the master bias from the science image
        return image_data - master_bias_data

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bias subtraction to animage represented as a numpy array.
        Parameters
        ----------
        image : np.ndarray
            The science image data to be bias-subtracted.
        Returns
        -------
        np.ndarray
            The bias-subtracted science image.
        Usage
        -----
         task = BiasSubtractionTask(master_bias=master_bias_array)
         result = task(science_image_array)
        """
        return self._subtract_bias(image)



