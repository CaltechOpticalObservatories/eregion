import copy
from typing import List, Optional, Iterable, Iterator
import numpy as np

from tasks.task import LazyTask
from datamodels.image import DetImage
from datamodels.image_utils import ensure_dataarray

from prefect import task, flow
from prefect.futures import wait


class BiasSubtraction(LazyTask):
    required_keys = ["master_bias"]
    depends_on = ["MasterBiasTask"]

    def __init__(self, name: Optional[str] = None, **kwargs):
        self.master_bias = kwargs["master_bias"]
        kwargs.pop("master_bias")
        super().__init__(name=name, **kwargs)

    def _bias_array(self) -> np.ndarray:
        if isinstance(self.master_bias, DetImage):
            return self.master_bias.data.values
        elif isinstance(self.master_bias, np.ndarray):
            return self.master_bias
        else:
            raise ValueError("master_bias must be either a DetImage or a numpy array.")

    @staticmethod
    def _subtract(image_data: np.ndarray, master_bias_data: np.ndarray) -> np.ndarray:
        return image_data - master_bias_data

    @task
    def _process_single_image(self, img: DetImage, master_bias_data: np.ndarray) -> DetImage:
        """
        Process a single DetImage by subtracting the master bias.
        :param img: DetImage
            The science image to be bias-subtracted.
        :return: DetImage
            The bias-subtracted science image.
        """
        if isinstance(img, DetImage):
            # initialize a new DetImage for the bias-subtracted result as a copy of the input image
            bias_subtracted = copy.deepcopy(img)
            bias_subtracted.data = ensure_dataarray(self._subtract(img.data.values, master_bias_data))
            bias_subtracted.meta["bias_subtracted"] = True
            bias_subtracted.image_type = f'bias_sub_{img.image_type}'
            return bias_subtracted
        else:
            raise ValueError("image must be an instance of DetImage.")

    @flow
    def lazy_run(
        self,
        images: Iterable[DetImage],
        batch_size: int = 1,
    ) -> Iterator[List[DetImage]]:
        """
        Lazy execution: yield processed batches as they complete.
        - images: an iterable/stream of DetImage
        - batch_size: number of images to process per batch
        """
        master_bias_data = self._bias_array()

        batch: List[DetImage] = []
        for img in images:
            batch.append(img)
            if len(batch) >= batch_size:
                futures = [self._process_single_image.submit(i, master_bias_data) for i in batch]
                wait(futures)
                yield [f.result() for f in futures]
                batch = []

        if batch:
            futures = [self._process_single_image.submit(i, master_bias_data) for i in batch]
            wait(futures)
            yield [f.result() for f in futures]

    @flow
    def run(self, images: list[DetImage]) -> list[DetImage]:
        """
        Subtract the master bias from the given images.
        :param images: list of DetImage
            List of science images to be bias-subtracted.
        :return: list of DetImage
            List of bias-subtracted science images.
        """
        master_bias_data = self._bias_array()
        # Process each image and return the list of bias-subtracted images
        futures = [self._process_single_image.submit(img, master_bias_data) for img in images]
        wait(futures)
        return [f.result() for f in futures]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Convenience non-Prefect path for raw arrays.
        """
        master_bias_data = self._bias_array()
        return self._subtract(image, master_bias_data)




