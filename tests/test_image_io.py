import numpy as np
import xarray as xr

from eregion.datamodels import DetectorProperties, DetImage, DetImageMeta, ImageBundle


def make_dataarray(shape=(10, 10)):
    arr = np.random.rand(*shape)
    return xr.DataArray(arr, dims=("y", "x"), coords={"y": np.arange(shape[0]), "x": np.arange(shape[1])})


def test_detimage_to_netcdf_appends_the_extension(tmp_path):
    meta = DetImageMeta(name="D1", properties=DetectorProperties(pixel_size=1.0, x_size=4, y_size=4),
                        focal_plane_position=None)
    img = DetImage(data=make_dataarray((4, 4)), meta=meta)
    for name in ("plain", "flat_0.5s", "done.nc"):
        img.to_netcdf(tmp_path / name)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["done.nc", "flat_0.5s.nc", "plain.nc"]


def test_imagebundle_save_and_load_keep_order(tmp_path):
    from tests.image_helpers import make_image
    bundle = ImageBundle([make_image(seed) for seed in range(12)])
    bundle.save(tmp_path / "bundle")
    loaded = ImageBundle.load(str(tmp_path / "bundle"))
    assert [img.data.values[0, 0] for img in loaded] == [img.data.values[0, 0] for img in bundle]
