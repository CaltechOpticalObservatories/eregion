import numpy as np

from tasks import Task
from datamodels import DetImage
from utils import ensure_dataarray, load_class


# Task to generate master bias
class MasterBias(Task):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def run(self, bias_images: list[DetImage]) -> dict[str,list[DetImage]]:
        """
        Generate master bias frames from a list of bias DetImage objects.
        :param bias_images: list of DetImage
            List of detector images containing bias frames.
        :return: {"master_biases": list[DetImage]}
            Dictionary with key 'master_biases' containing a list of master bias DetImage objects.
        """
        # Check that bias_images are DetImage instances
        if not isinstance(bias_images, list) or not all(isinstance(img, DetImage) for img in bias_images):
            raise ValueError("bias_images must be a list of DetImage instances.")
        # Verify that all input images are of 'bias' image_type
        # TODO: Add a init kwarg to specify the expected image_type of input bias frames, and check against that instead of hardcoding 'bias'
        for img in bias_images:
            if 'bias' not in img.image_type.lower():
                raise ValueError(f"All input images must have image_type 'bias'. Found '{img.image_type}' in {img.meta.filename}.")

        # Group bias images by detector name
        bias_dict = {}
        for img in bias_images:
            det_name = img.meta.name if 'name' in img.meta else 'default'
            if det_name not in bias_dict:
                bias_dict[det_name] = []
            bias_dict[det_name].append(img)

        # Create master bias for each detector
        master_biases = []
        for det_name, imgs in bias_dict.items():
            # Initialize master bias DetImage
            master_bias = DetImage(image_type="master_bias")
            # Combine bias data
            bias_data = [img.data.values for img in imgs]
            master_bias.data = ensure_dataarray(self._create_masterbias(bias_data,
                                                                        method=self.meta.get('method', 'median')))
            # Copy metadata and outputs from the first bias image
            master_bias.meta = imgs[0].meta
            master_bias.focal_plane = imgs[0].focal_plane
            master_bias.meta.update({'filenames':', '.join([img.meta.filename for img in imgs])})
            master_bias.outputs.update(imgs[0].outputs)
            master_biases.append(master_bias)
        return {'master_biases': master_biases}

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
        self.set_method(method)
        method_cls = load_class(self.methods[method])
        return method_cls(biases)

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

