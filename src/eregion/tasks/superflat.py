from copy import deepcopy
import numpy as np
import xarray as xr
from typing import Callable

from eregion.tasks import Task
from eregion.datamodels import DetImage, ImageBundle
from eregion.tasks.imagegen import ImageResult
from eregion.core.image_operations import welford_update, welford_finalize, stochastic_median_update


class SuperCombine(Task):
    """
    Lazy loading and combine method for combining large number of images into a single image.
    """
    task_result = ImageResult

    # running sums kept in self.dataset for each method
    accumulators = {
        'mean': ['sum', 'sum_of_squares', 'count', 'variance'],
        'welford': ['count', 'mean', 'm2', 'median', 'variance'],
    }

    def __init__(self, name=None, **kwargs):
        """
        :param name: str, task name
        :keyword method: str, method name 'mean' (running sums) or 'welford' (Welford mean/std plus a
            Robbins-Monro streaming median approximation stored in the 'median' plane)

        Attributes:

        - self.dataset: xarray.Dataset, holds the relevant sums as images are combined
        - self.superimage: DetImage, final supercombined image, its outputs and meta are set from the first image in the series
        """
        kwargs['method'] = kwargs.get('method', 'mean')
        super().__init__(name, **kwargs)
        self.dataset: xr.Dataset | None = None
        self.superimage: DetImage | None = None

    def _add_image(self, image: DetImage) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize self.dataset and self.superimage from the first image, add the input variance (if the image has a
        'std' plane) of the non-masked pixels to the 'variance' data_var, then unload the image data.

        :param image: DetImage to combine
        :return: tuple of (image data, valid pixel boolean array)
        """
        ims = image.get_data('all')
        mask = image.get_mask('combined')
        valid = np.ones(ims['data'].shape, dtype=bool) if mask is None else ~mask.values.astype(bool)

        if self.dataset is None:
            self.dataset = xr.Dataset(data_vars={key: xr.zeros_like(ims['data'], dtype=float)
                                                 for key in self.accumulators[self.method_name]})
            # copy outputs without parent/masks so the first image keeps its own outputs and masks
            outputs = {oid: out.model_copy(update={'parent': None, 'masks': None})
                       for oid, out in image.outputs.items()}
            self.superimage = DetImage(meta=deepcopy(image.meta), output_objects=outputs)
            self.superimage.image_type['supercombined'] = True
            self.superimage.meta['num_images'] = 0
        elif ims['data'].shape != self.dataset['count'].shape:
            raise ValueError(f"Image {image.id} has shape {ims['data'].shape}, "
                             f"expected {self.dataset['count'].shape} like the previous images.")

        data = ims['data'].values
        if 'std' in ims.data_vars:
            self.dataset['variance'].values += np.where(valid, ims['std'].values ** 2, 0.0)

        # update total images count in superimage meta, unload image data to free up memory
        self.superimage.meta['num_images'] += 1
        image.unload()
        return data, valid

    def mean_combine(self, image: DetImage):
        """
        Adds the non-masked image data to the 'sum' data_var, adds its square to the 'sum_of_squares' data_var.
        Adds the non-masked pixels into the total counts tracking data_var 'count'.
        If input image already has a 'std' plane then squares it and adds it to the 'variance' data_var.

        :param image: DetImage to combine
        :return: None
        """
        data, valid = self._add_image(image)
        ma_im = np.where(valid, data, 0.0)
        self.dataset['sum'].values += ma_im
        self.dataset['sum_of_squares'].values += (ma_im ** 2)
        self.dataset['count'].values += valid

    def welford_combine(self, image: DetImage):
        """
        Updates the Welford 'count', 'mean' and 'm2' data_vars with the non-masked image pixels, then updates the
        streaming 'median' estimate using the running std.
        If input image already has a 'std' plane then squares it and adds it to the 'variance' data_var.

        :param image: DetImage to combine
        :return: None
        """
        data, valid = self._add_image(image)
        count = self.dataset['count'].values
        m2 = self.dataset['m2'].values
        welford_update(count, self.dataset['mean'].values, m2, data, valid)
        running_std = np.sqrt(np.divide(m2, count - 1, out=np.zeros_like(m2), where=count > 1))
        stochastic_median_update(self.dataset['median'].values, count, running_std, data, valid)

    def _finalize(self) -> xr.Dataset:
        """
        Build the output planes from the accumulators, without modifying them so this can be called after every batch.

        :return: xr.Dataset with 'data', 'std', 'count', 'variance' (and 'median' for the welford method)
        """
        ds = self.dataset
        counts = ds['count']
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.method_name == 'welford':
                stats = welford_finalize(counts.values, ds['mean'].values, ds['m2'].values)
                stats['data'] = stats.pop('mean')
                out = xr.Dataset({key: counts.copy(data=val) for key, val in stats.items()})
                out['median'] = ds['median'].where(counts > 0)
            else:
                out = xr.Dataset({'data': (ds['sum'] / counts).where(counts > 0)})
                sq_dev = (ds['sum_of_squares'] - ds['sum'] ** 2 / counts).clip(min=0)
                out['std'] = (sq_dev / (counts - 1)).where(counts > 1) ** 0.5
                out['count'] = counts.copy()
            out['variance'] = (ds['variance'] / counts ** 2).where(counts > 0)
        return out

    @property
    def final_image_result(self) -> ImageResult:
        if self.dataset is None:
            raise AttributeError("No combined superimage to return!!!")
        if self.superimage.num_outputs == 0:
            self.logger.warning('No outputs found in superimage DetImage instance, is that expected?')

        self.superimage.set_data(self._finalize())
        return ImageResult(data=[self.superimage])

    def run(self, images: ImageBundle) -> ImageResult:
        images = images if isinstance(images, ImageBundle) else ImageBundle(images)

        for image in images:
            self.method(image)

        return self.final_image_result

    @property
    def methods(self) -> dict[str, Callable]:
        return {
            'mean': self.mean_combine,
            'welford': self.welford_combine
        }
