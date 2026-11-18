"""
Small synthetic detector images shared by the datamodel and task tests.
"""
import numpy as np
import xarray as xr

from eregion.datamodels import CCDOutput, DetectorProperties, DetImage, DetImageMeta, FocalPlanePosition

SHAPE = (8, 12)


def make_outputs():
    # Two side-by-side amplifiers, each with a 1-pixel prescan/overscan border
    return {
        "A": CCDOutput(id="A", ext_id=0, ext_slice=(slice(0, 8), slice(0, 6)), data_slice=(slice(0, 8), slice(0, 6)),
                       serial_prescan=slice(0, 1), serial_overscan=slice(5, 6),
                       parallel_prescan=slice(0, 1), parallel_overscan=slice(7, 8)),
        "B": CCDOutput(id="B", ext_id=0, ext_slice=(slice(0, 8), slice(6, 12)), data_slice=(slice(0, 8), slice(6, 12)),
                       serial_prescan=slice(6, 7), serial_overscan=slice(11, 12),
                       parallel_prescan=slice(0, 1), parallel_overscan=slice(7, 8)),
    }


def make_image(seed=0, name="D1", x_cen=6.0, std=2.0, mask_pixel=None):
    rng = np.random.default_rng(seed)
    coords = {"y": np.arange(SHAPE[0]), "x": np.arange(SHAPE[1])}
    data = xr.Dataset({
        "data": (("y", "x"), rng.normal(100, 5, SHAPE)),
        "std": (("y", "x"), np.full(SHAPE, std)),
    }, coords=coords)
    meta = DetImageMeta(name=name, properties=DetectorProperties(pixel_size=1.0, x_size=SHAPE[1], y_size=SHAPE[0]),
                        focal_plane_position=FocalPlanePosition(x_cen=x_cen, y_cen=4.0))
    img = DetImage(data=data, output_objects=make_outputs(), meta=meta)
    if mask_pixel is not None:
        out = img.outputs["A"]
        mask = xr.zeros_like(out.data, dtype=bool)
        mask.loc[{"y": mask_pixel[0], "x": mask_pixel[1]}] = True
        out.set_mask("cr", mask)
    return img
