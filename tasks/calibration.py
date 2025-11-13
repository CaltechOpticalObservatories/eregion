from tasks.task import Task
import numpy as np
from datamodels.image import DetImage, Output
from utils.image_operations import median_combine, mean_combine

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
        bias_data = [bias_image.data for bias_image in bias_images]
        master_bias.data = self._create_masterbias(bias_data,
                                                   method=self.meta['method'] if 'method' in self.meta else 'median')
        # Copy metadata and outputs from the first bias image
        master_bias.meta = bias_images[0].meta.copy()
        master_bias.focal_plane = bias_images[0].focal_plane.copy()
        master_bias.meta["filenames"] = ', '.join([img.meta.get("filename", "unknown") for img in bias_images])
        for output in bias_images[0].outputs:
            op = output.copy()
            op.filename = ''
            master_bias.add_output(op)
        return master_bias

    def _create_masterbias(self, biases: list[np.ndarray], method='median') -> np.ndarray:
        if method == 'median':
            return median_combine(biases)
        else:
            raise NotImplementedError

    def __call__(self, biases: list[np.ndarray]):
        """
        Generate a master bias frame from a list of bias images.
        :param biases: list of numpy arrays
            List of detector images containing bias frames.
        :return: master_bias: numpy array
            The generated master bias frame.
        """
        return self._create_masterbias(biases)