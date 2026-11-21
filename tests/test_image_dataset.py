"""
DetImage keeps data, mask and label vars in one Dataset; these tests cover naming, writing, loading and files.
"""
import json
import pickle

import numpy as np
import pytest
import xarray as xr
from astropy.io import fits

from eregion.datamodels import (
    CCDOutput,
    DetectorProperties,
    DetImage,
    DetImageMeta,
    FocalPlaneImage,
    Output,
)
from tests.image_helpers import SHAPE, make_image


######################################## names and checks ########################################
@pytest.mark.parametrize("name, values, error, match", [
    ("x_mask", np.zeros(SHAPE), TypeError, "must be bool"),
    ("x_label", np.zeros(SHAPE, dtype=bool), TypeError, "must be integer"),
    ("_mask", np.zeros(SHAPE, dtype=bool), ValueError, "<prefix>_mask"),
    ("combined", np.zeros(SHAPE), ValueError, "reserved"),
    (3, np.zeros(SHAPE), TypeError, "must be str"),
    ("std", np.zeros((2, 2)), ValueError, "shape"),
])
def test_set_var_checks_names_dtypes_and_shapes(name, values, error, match):
    with pytest.raises(error, match=match):
        make_image().set_var(name, values)


def test_whole_var_with_other_coords_is_rejected_instead_of_nan_filled():
    img = make_image()
    shifted = xr.DataArray(np.zeros(SHAPE), dims=("y", "x"),
                           coords={"y": np.arange(SHAPE[0]) + 1, "x": np.arange(SHAPE[1])})
    with pytest.raises(ValueError, match="coords"):
        img.set_var("other", shifted)
    assert "other" not in img.data_names


def test_combine_is_for_masks_only():
    with pytest.raises(ValueError, match="combine=True ORs masks"):
        make_image().set_var("data", np.zeros(SHAPE), combine=True)


def test_get_data_vars_and_kind_specific_reads():
    img = make_image(mask_pixel=(2, 2))
    assert img.data_names == ["data", "std"] and img.mask_names == ["cr_mask"] and img.label_names == []
    with pytest.raises(KeyError, match="no var 'nope'"):
        img.get_var("nope")
    with pytest.raises(ValueError, match="takes a data var"):
        img.get_masked("cr_mask")
    assert img.get_mask(None).dtype == bool and not img.get_mask(None).any()


######################################## region writes ########################################
def test_region_write_creates_missing_vars_per_kind():
    img = make_image()
    region = (slice(0, 2), slice(0, 3))
    img.set_var("flat", np.ones((2, 3)), region)
    img.set_var("new_mask", np.ones((2, 3), dtype=bool), region)
    img.set_var("new_label", np.ones((2, 3), dtype=np.int32), region)
    assert np.isnan(img.get_var("flat").values[3, 3]) and img.get_var("flat").values[0, 0] == 1
    assert img.get_var("new_mask").dtype == bool and int(img.get_var("new_mask").sum()) == 6
    assert img.get_var("new_label").values[5, 5] == 0
    with pytest.raises(ValueError, match="integer data var"):
        img.set_var("counts", np.ones((2, 3), dtype=int), region)


def test_region_write_checks_shape_dtype_and_coords():
    img = make_image()
    region = {"y": slice(0, 2), "x": slice(0, 3)}
    with pytest.raises(ValueError, match="shape"):
        img.set_var("data", np.ones((3, 3)), region)
    img.set_var("counts", np.zeros(SHAPE, dtype=np.int64))
    with pytest.raises(TypeError, match="truncate"):
        img.set_var("counts", np.full((2, 3), 0.5), region)
    wrong = xr.DataArray(np.ones((2, 3)), dims=("y", "x"), coords={"y": [5, 6], "x": [0, 1, 2]})
    with pytest.raises(ValueError, match="region's coords"):
        img.set_var("data", wrong, region)


def test_reversed_region_round_trip():
    img = make_image()
    region = {"y": slice(3, 0, -1), "x": slice(0, 2)}  # rows 3, 2, 1
    block = img.get_var("data", region).copy()  # get_var may return a view of the image
    assert list(block["y"].values) == [3, 2, 1]
    img.set_var("data", block.values + 100, region)
    assert np.allclose(img.get_var("data", region), block + 100)
    # a DataArray in ascending order does not match the reversed region
    with pytest.raises(ValueError, match="region's coords"):
        img.set_var("data", block.sortby("y"), region)


def test_set_vars_and_remove_vars():
    img = make_image(mask_pixel=(2, 2))
    img.set_vars({"count": np.ones(SHAPE, dtype=np.int64), "bad_mask": np.zeros(SHAPE, dtype=bool)})
    assert "count" in img.data_names and "bad_mask" in img.mask_names
    img.remove_vars("count", "bad_mask", "cr_mask")
    assert img.mask_names == [] and "count" not in img.data_names
    with pytest.raises(ValueError, match="unload"):
        img.remove_vars("data")
    with pytest.raises(KeyError):
        img.remove_vars("missing_mask")


######################################## loading ########################################
def _lazy_image():
    def loader(_):
        return [np.arange(16.0).reshape(4, 4)], [fits.Header({"T": 1})]
    output = Output(id="A", ext_id=0, ext_slice=(slice(0, 4), slice(0, 4)), data_slice=(slice(0, 4), slice(0, 4)))
    meta = DetImageMeta(name="D1", filename="x.fits",
                        properties=DetectorProperties(pixel_size=1.0, x_size=4, y_size=4), focal_plane_position=None)
    return DetImage(output_objects={"A": output}, meta=meta, loader=loader)


def test_masks_set_before_load_survive_loading_and_unloading():
    img = _lazy_image()
    img.outputs["A"].set_var("early_mask", np.eye(4, dtype=bool))
    assert not img.is_loaded and img.mask_names == ["early_mask"]
    assert img.data.values[3, 3] == 15 and img.is_loaded
    img.unload()
    assert not img.is_loaded and int(img.get_mask().sum()) == 4
    assert img.get_masked().mask.sum() == 4  # reloads the data


def test_unload_and_load_need_a_loader_and_filename():
    with pytest.raises(ValueError, match="no loader"):
        make_image().unload()
    img = _lazy_image()
    img.meta.filename = None
    with pytest.raises(ValueError, match="filename"):
        img.load()
    with pytest.raises(TypeError, match="loader="):
        DetImage(data=lambda f: None)


######################################## outputs ########################################
def test_output_without_parent_and_masks_argument():
    orphan = Output(id="o", ext_id=0, ext_slice=(slice(0, 2), slice(0, 2)), data_slice=(slice(0, 2), slice(0, 2)))
    with pytest.raises(ValueError, match="no parent"):
        orphan.get_var()
    with pytest.raises(ValueError, match="no parent"):
        orphan.set_var("x_mask", np.zeros((2, 2), dtype=bool))
    with pytest.raises(TypeError, match="hold no masks"):
        Output(id="o", ext_id=0, ext_slice=(slice(0, 2),), data_slice=(slice(0, 2),), masks=None)


def test_copy_writes_no_masks_field_into_output_json():
    img = make_image(mask_pixel=(2, 2)).copy()
    assert "masks" not in json.loads(img.outputs["A"].to_json(exclude={"parent"}))


def test_pickle_round_trip_keeps_parent_links_and_masks():
    img = make_image(mask_pixel=(2, 2))
    back = pickle.loads(pickle.dumps(img))
    assert all(out.parent is back for out in back.outputs.values())
    assert back.get_mask("cr_mask").equals(img.get_mask("cr_mask"))


######################################## focal plane ########################################
def test_focal_plane_mosaics_every_kind():
    tiles = [make_image(0, name="D1", x_cen=6.0, mask_pixel=(2, 2)), make_image(1, name="D2", x_cen=18.0)]
    labels = xr.zeros_like(tiles[1].data, dtype=np.int32)
    labels[1, 1] = 4
    tiles[1].set_var("cosmic_ray_label", labels)
    fp = FocalPlaneImage(num_detectors=2, det_images=tiles)
    assert fp.dataset["cr_mask"].dtype == bool and int(fp.get_mask().sum()) == 1
    assert fp.dataset["cosmic_ray_label"].dtype == np.int32 and int(fp.get_var("cosmic_ray_label").max()) == 4
    assert fp.label_names == ["cosmic_ray_label"] and fp.get_masked().mask.sum() == 1


######################################## netcdf ########################################
def test_netcdf_round_trip_keeps_kinds_and_clears_attrs(tmp_path):
    img = make_image(mask_pixel=(2, 2))
    img.set_var("cosmic_ray_label", np.ones(SHAPE, dtype=np.int32))
    img.to_netcdf(tmp_path / "img")
    assert xr.load_dataset(tmp_path / "img.nc").attrs["eregion_format"] == 2
    loaded = DetImage.from_netcdf(tmp_path / "img.nc")
    assert loaded.get_var("cr_mask").dtype == bool
    assert loaded.dataset.attrs == {}
    assert loaded.dataset.equals(img.dataset)
    assert isinstance(loaded.outputs["A"], CCDOutput)


def _write_legacy_file(path, extra_vars=None):
    """ A file in the layout written before eregion_format 2: float masks with NaN, attrs listing the keys. """
    img = make_image()
    mask = np.full(SHAPE, np.nan)
    mask[:, :6] = 0.0
    mask[2, 2] = 1.0
    labels = np.zeros(SHAPE, dtype=np.int32)
    labels[3, 3] = 2
    ds = img.dataset.copy()
    ds["cr"] = (("y", "x"), mask)
    ds["cosmic_ray_labels"] = (("y", "x"), labels)
    for name, values in (extra_vars or {}).items():
        ds[name] = (("y", "x"), values)
    ds.attrs = {
        "mask_keys": json.dumps(["cr"]),
        "label_keys": json.dumps(["cosmic_ray_labels"]),
        "meta": img.meta.to_json(),
        "image_type": json.dumps(img.image_type),
        "outputs": json.dumps({k: o.to_json(exclude={"parent"}) for k, o in img.outputs.items()}),
        "output_class": "CCDOutput",
    }
    ds.to_netcdf(path)


def test_legacy_netcdf_is_converted_and_resaved_in_the_new_format(tmp_path, caplog):
    _write_legacy_file(tmp_path / "old.nc")
    old = DetImage.from_netcdf(tmp_path / "old.nc")
    assert old.mask_names == ["cr_mask"] and old.label_names == ["cosmic_ray_label"]
    assert old.get_var("cr_mask").dtype == bool and int(old.get_mask().sum()) == 1
    assert int(old.get_var("cosmic_ray_label").values[3, 3]) == 2
    assert "renamed" in caplog.text

    old.to_netcdf(tmp_path / "new")
    again = DetImage.from_netcdf(tmp_path / "new.nc")
    assert again.dataset.equals(old.dataset)


def test_legacy_netcdf_with_an_unlisted_suffixed_var_raises(tmp_path):
    _write_legacy_file(tmp_path / "old.nc", extra_vars={"stray_mask": np.zeros(SHAPE)})
    with pytest.raises(ValueError, match="stray_mask"):
        DetImage.from_netcdf(tmp_path / "old.nc")
