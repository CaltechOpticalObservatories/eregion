import numpy as np
import xarray as xr

from eregion.datamodels import DetImage, CCDOutput
from eregion.tasks.preprocessing import VignetteMasking


def test_vignette_masking_only_masks_image_region():
    ny, nx = 220, 230
    rng = np.random.default_rng(1)
    arr = rng.normal(0, 5, (ny, nx))
    # image region is y in [10, 210), x in [10, 220); prescan/overscan stay near zero
    yy, xx = np.indices((200, 210))
    illum = np.full((200, 210), 1000.0)
    illum[np.hypot(yy, xx) < 50] = 200.0  # vignetted corner
    arr[10:210, 10:220] += illum
    data = xr.DataArray(arr, dims=("y", "x"), coords={"y": np.arange(ny), "x": np.arange(nx)})
    ccd = CCDOutput(
        id="C",
        ext_id=0,
        ext_slice=(slice(0, ny), slice(0, nx)),
        data_slice=(slice(0, ny), slice(0, nx)),
        serial_prescan=slice(0, 10),
        serial_overscan=slice(220, 230),
        parallel_prescan=slice(0, 10),
        parallel_overscan=slice(210, 220),
        parallel_axis="y",
    )
    det = DetImage(data=data, output_objects={"C": ccd})
    det.image_type["bias_subtracted"] = True

    out = VignetteMasking()._process_single_image(det)
    mask = out.outputs["C"].masks["vignette_mask"].values

    assert mask.shape == (ny, nx)
    assert mask[10:30, 10:30].all()
    assert not mask[100:210, 100:220].any()
    # prescan/overscan never masked
    assert not mask[:10].any() and not mask[210:].any()
    assert not mask[:, :10].any() and not mask[:, 220:].any()
    assert out.image_type["vignette_masked"]
