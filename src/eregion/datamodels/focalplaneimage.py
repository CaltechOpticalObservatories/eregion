from typing import Optional
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from eregion.utils import configure_logger
from eregion.datamodels import DetImage, DetImageMeta, ImageBundle
from eregion.core.image_operations import flip_and_rotate

logger = configure_logger(__name__)

class FocalPlaneImage:
    """
    Composite focal-plane image made by placing multiple DetImage tiles.

    Parameters
    ----------
    num_detectors: int
        Number of detector tiles expected.
    dim: tuple[float, float]
        Dimensions of the focal plane image (height, width) in mm.
    det_images: Optional[List[DetImage]]
        List of DetImage objects to place on the focal plane.

    Attributes
    ----------
    meta: dict
        Metadata dictionary for the focal plane image.
    num_detectors: int
        Number of detector tiles expected.
    dim_mm: Optional[tuple[float, float]]
        Dimensions of the focal plane image (height, width) in mm.
    pixel_size: Optional[float]
        Pixel size in mm, assumed to be the same for all DetImage tiles.
    data: Optional[xr.Dataset]
        xarray Dataset holding the composite focal-plane image data.
    masks: Optional[xr.Dataset]
        xarray Dataset holding the composite focal-plane image masks.
    table: Optional[pd.DataFrame]
        DataFrame keeping track of DetImage data positions within the focal-plane data array.
    det_images: ImageBundle
        Bundle of DetImage objects placed on the focal plane.
    """

    def __init__(
        self,
        num_detectors: int,
        dim: Optional[tuple[float, float]] = None,
        det_images: Optional[list[DetImage] | ImageBundle] = None,
        **kwargs,
    ):
        self.meta: dict = {}
        if kwargs:
            self.meta.update(kwargs)

        self.num_detectors = int(num_detectors)
        self.dim_mm = tuple(dim) if dim is not None else None
        self.pixel_size = None
        self._data: Optional[xr.Dataset] = None   # To hold det image data in one array
        self.masks: Optional[xr.Dataset] = None  # To hold det image masks in one dataset
        self.table: Optional[pd.DataFrame] = None  # To keep track of det_image data position within focal-plane data array
        self.det_images: ImageBundle = ImageBundle()  # To hold det images

        if det_images is not None:
            det_images = det_images if isinstance(det_images, ImageBundle) else ImageBundle(det_images)
            self.update(det_images)

    def update(self, det_images: ImageBundle) -> None:
        self.add_det_images(det_images)
        self.construct_focal_plane_image()

    def add_det_images(self, det_images: ImageBundle):
        for det_image in det_images:
            if len(self.det_images) == self.num_detectors:
                logger.error(
                    f"Number of DetImages added have reached number of detectors present in this focal-plane.")
                break
            self.validate_det_image(det_image)
            self.det_images.append(det_image)

    def validate_det_image(self, det_image: DetImage) -> None:
        """
        Accepts either validated DetImageMeta or legacy dict with required keys.
        """
        if not (isinstance(det_image.meta, DetImageMeta) or isinstance(det_image.meta, dict)):
            raise ValueError("DetImage.meta must be DetImageMeta or dict.")
        required = {"properties", "focal_plane_position"}
        if not required.issubset(det_image.meta.keys()):
            raise ValueError("DetImage.meta missing required keys for focal-plane placement.")
        # attempt to coerce for consistent downstream access
        det_image.meta = DetImageMeta.model_validate(det_image.meta)

        # check pixel size is same for all det images
        pixsize = det_image.meta.properties.pixel_size
        if self.pixel_size is None:
            self.pixel_size = pixsize
        elif pixsize != self.pixel_size:
            raise ValueError("All DetImage objects must have the same pixel_size for focal-plane assembly.")

        # check that masks have been built
        if det_image.masks is None:
            det_image.build_full_mask()

    def construct_focal_plane_image(self):
        if len(self.det_images) == 0:
            raise ValueError("No DetImage objects to assemble.")

        frames = []
        for det_image in self.det_images:
            props = det_image.meta.properties  # type: ignore[union-attr]
            pos = det_image.meta.focal_plane_position  # type: ignore[union-attr]
            xhalf, yhalf = props.x_size / 2, props.y_size / 2
            frames.append(
                {
                    "det_id": det_image.id,  # type: ignore[arg-type]
                    "x_min": int(pos.x_cen / self.pixel_size - xhalf),
                    "x_max": int(pos.x_cen / self.pixel_size - xhalf) + props.x_size,
                    "y_min": int(pos.y_cen / self.pixel_size - yhalf),
                    "y_max": int(pos.y_cen / self.pixel_size - yhalf) + props.y_size,
                    "angle": pos.angle if hasattr(pos, "angle") else None,
                    "flip_x": pos.flip_x if hasattr(pos, "flip_x") else None,
                    "flip_y": pos.flip_y if hasattr(pos, "flip_y") else None,
                }
            )
        frames_df = pd.DataFrame(frames)

        # Verify that there are no overlapping detectors, i.e. area covered inside corners should not overlap
        for i in range(len(frames_df)):
            for j in range(i + 1, len(frames_df)):
                a = frames_df.iloc[i]
                b = frames_df.iloc[j]
                separated = (a["x_max"] <= b["x_min"]) or (a["x_min"] >= b["x_max"]) or \
                            (a["y_max"] <= b["y_min"]) or (a["y_min"] >= b["y_max"])
                if not separated:
                    raise ValueError(f"Detectors {a['det_id']} and {b['det_id']} overlap in focal plane.")

        # Verify the size of the focal plane image
        calc_dim = np.array([frames_df["y_max"].max() - frames_df["y_min"].min(),
                            frames_df["x_max"].max() - frames_df["x_min"].min()])
        if self.dim_mm is not None:
            dim_pix = (np.array(self.dim_mm) / self.pixel_size).astype(int)
            if any(calc_dim > dim_pix):
                logger.warning("Provided dim %s < computed dim %s.", dim_pix, calc_dim)
        else:
            dim_pix = calc_dim.astype(int)

        # Initialize DataArray
        coords = {"y": np.arange(frames_df["y_min"].min(), frames_df["y_max"].max(), 1),
                  "x": np.arange(frames_df["x_min"].min(), frames_df["x_max"].max(), 1)}
        dataarr = xr.DataArray(data=np.zeros(dim_pix, dtype=float), coords=coords, dims=["y", "x"])
        self._data = xr.Dataset(data_vars={'data': dataarr.copy(deep=True)})
        maskarr = xr.DataArray(data=np.zeros(dim_pix, dtype=bool), coords=coords, dims=["y", "x"])
        self.masks = xr.Dataset(data_vars={'sigma_clip_mask': maskarr.copy(deep=True)})

        # Place tiles
        for i in range(len(frames_df)):
            row = frames_df.iloc[i]
            slc = {'y': slice(row['y_min'], row['y_max'] - 1), 'x': slice(row['x_min'], row['x_max'] - 1)}
            di = self.det_images[i]
            didataset = di.get_data('all')
            if didataset is None:
                raise ValueError(f"DetImage at index {i} has no data.")
            else:
                data_vars = list(didataset.data_vars)
                for v in data_vars:
                    if v not in self._data.data_vars:
                        self._data[v] = dataarr.copy(deep=True)
                    self._data[v].loc[slc] = flip_and_rotate(didataset[v].values, angle=row['angle'], flip_x=row['flip_x'],
                                                           flip_y=row['flip_y'])

                mask_keys = list(di.masks.data_vars) if di.build_full_mask() else []
                dimasks = di.masks
                for m in mask_keys:
                    if m not in self.masks.data_vars:
                        self.masks[m] = maskarr.copy(deep=True)
                    self.masks[m].loc[slc] = flip_and_rotate(dimasks[m].values, angle=row['angle'], flip_x=row['flip_x'],
                                           flip_y=row['flip_y'])

        self.table = frames_df

    @property
    def data(self) -> xr.DataArray:
        if self._data is None:
            raise ValueError("Focal-plane image data has not been constructed yet.")
        return self._data['data']

    def show(self, ax=None, save=None, show_det_id=False, with_mask=False, mask_key='sigma_clip_mask',
             data_var:str = 'data', textcolor="yellow", **imshow_kwargs):
        """
        Plot the focal-plane image with optional detector boundaries and masks.
        :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        :param save: Path to save the figure. If None, the figure will not be saved.
        :param show_det_id: bool, whether to show detector IDs on the plot.
        :param with_mask: bool, whether to overlay the mask on the image specified by mask_key.
        :param mask_key: str, key of the mask to overlay on the image. Default is 'sigma_clip_mask'.
        :param data_var: str, key of the data variable to plot. Default is 'data'.
        :param textcolor: str, color of the text to display on the plot. Default is "yellow".
        :param imshow_kwargs: Additional keyword arguments to pass to the xarray.DataArray.plot.imshow function.
        :return: Matplotlib Axes object with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(1,1, figsize=(8, 8), tight_layout=True)
        # overlay mask if requested
        temp = self._data[data_var]
        if with_mask:
            if self.masks is not None and mask_key in self.masks:
                arr = np.ma.masked_array(data=temp.values, mask=self.masks[mask_key].values)
                arr = arr.filled(0)
                temp = xr.DataArray(data=arr, coords=temp.coords, dims=temp.dims)
            else:
                logger.warning("Mask key '%s' not found in masks. Showing unmasked data.", mask_key)
        im = temp.plot.imshow(ax=ax, **imshow_kwargs)

        # Draw detector boundaries
        if hasattr(self, "table"):
            for _, row in self.table.iterrows():
                rect = plt.Rectangle(
                    (row["x_min"], row["y_min"]),
                    row["x_max"] - row["x_min"],
                    row["y_max"] - row["y_min"],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
                if show_det_id:
                    ax.text(row["x_min"] + 150, row["y_min"] + 150, str(row["det_id"]), color=textcolor, fontsize=10)
        if save is not None:
            ax.figure.savefig(save)
        return ax

    # TODO: Add to_netcdf() and from_netcdf() methods


class FPImageBundle(ImageBundle[FocalPlaneImage]):
    """
    To store list of FocalPlaneImage objects.
    """
    image_class = FocalPlaneImage

    def _tabulate(self):
        tab = []
        for fpimage in self.images:
            imtype = {'object': fpimage, 'filename': fpimage.meta.get('filename', None)}
            imtype |= fpimage.det_images[0].image_type
            tab.append(imtype)
        self.list = pd.DataFrame(tab)
