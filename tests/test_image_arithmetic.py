
import numpy as np
import pytest
import xarray as xr

from eregion.core.image_operations import arithmetic, propagate_std
from eregion.core.image_stats import STATFUNCS, as_masked_array, do_statistics
from eregion.datamodels import CCDOutput, DetImage, FocalPlaneImage, Output
from tests.image_helpers import SHAPE, make_image

################################################ core ################################################
def test_propagate_std_known_values():
    a, b = np.array([4.0]), np.array([2.0])
    sa, sb = np.array([0.3]), np.array([0.4])
    assert np.allclose(propagate_std("add", a, b, sa, sb), 0.5)
    assert np.allclose(propagate_std("subtract", a, b, sa, sb), 0.5)
    assert np.allclose(propagate_std("multiply", a, b, sa, sb), np.hypot(2 * 0.3, 4 * 0.4))
    assert np.allclose(propagate_std("divide", a, b, sa, sb), np.hypot(0.3 / 2, 4 * 0.4 / 4))
    # scalar operand: std scales by |k| for multiply/divide
    assert np.allclose(propagate_std("multiply", a, -3.0, sa), 0.9)
    assert np.allclose(propagate_std("divide", a, 3.0, sa), 0.1)
    assert propagate_std("add", a, 1.0) is None


def test_arithmetic_validates_op_and_shape():
    a = np.ones((3, 4))
    assert arithmetic(a, np.full((3, 1), 2.0), "subtract").shape == (3, 4)
    with pytest.raises(ValueError, match="Unknown arithmetic operation"):
        arithmetic(a, 1, "power")
    with pytest.raises(ValueError, match="does not broadcast"):
        arithmetic(a, np.ones((4, 3)), "add")


def test_as_masked_array_casts_mask_to_bool():
    data = np.array([[1.0, np.nan], [3.0, 4.0]])
    ma = as_masked_array(data, mask=np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert ma.mask.dtype == bool and ma.mask.sum() == 1
    assert as_masked_array(data, mask_invalid=True).mask.sum() == 1


############################################### Output ###############################################
def test_output_scalar_op_writes_into_parent_and_scales_std():
    img = make_image()
    orig = img.dataset.copy(deep=True)
    img.outputs["A"].multiply(3)
    assert np.allclose(img.outputs["A"].data, orig["data"][:, :6] * 3)
    assert np.allclose(img.get_var("std")[:, :6], 6.0)
    # output B is untouched
    assert np.array_equal(img.outputs["B"].data, orig["data"][:, 6:])
    assert np.allclose(img.get_var("std")[:, 6:], 2.0)


def test_output_region_image_leaves_scans_untouched():
    img = make_image()
    out = img.outputs["A"]
    before = out.data.copy(deep=True)
    out.subtract(10, region="image")
    assert np.allclose(out.get_var(region="image"), before[1:7, 1:5] - 10)
    assert np.array_equal(out.get_var(region="serial_overscan"), before.sel(y=slice(1, 6), x=slice(5, 5)))
    assert np.array_equal(out.get_var(region=out.region_slicer("parallel_prescan", corner=True)),
                          before.sel(y=slice(0, 0)))


def test_output_get_region_names_and_errors():
    out = make_image().outputs["A"]
    assert out.region_names == ["full", "image", "serial_prescan", "serial_overscan",
                                "parallel_prescan", "parallel_overscan"]
    assert out.get_var(region="serial_overscan").shape == (6, 1)
    with pytest.raises(ValueError, match="Unknown region 'corner'"):
        out.get_var(region="corner")
    with pytest.raises(ValueError, match="scan regions only"):
        out.region_slicer("image", corner=True)


def test_output_scalar_broadcast_array_like_scan_subtraction():
    img = make_image()
    out = img.outputs["A"]
    before = out.data.values.copy()
    row_median = np.median(before, axis=1, keepdims=True)
    out.subtract(row_median, propagate_std=False)
    assert np.allclose(out.data.values, before - row_median)


def test_output_binary_op_values_std_and_masks():
    a, b = make_image(0, mask_pixel=(2, 2)), make_image(1, mask_pixel=(3, 3))
    expected = a.outputs["A"].data.values - b.outputs["A"].data.values
    a.outputs["A"].subtract(b.outputs["A"])
    assert np.allclose(a.outputs["A"].data.values, expected)
    assert np.allclose(a.get_var("std")[:, :6], np.hypot(2.0, 2.0))
    full_mask = a.get_mask("cr_mask")
    assert bool(full_mask.sel(y=2, x=2)) and bool(full_mask.sel(y=3, x=3)) and int(full_mask.sum()) == 2


def test_output_binary_op_flags_turn_off_masks_and_std():
    a, b = make_image(0, mask_pixel=(2, 2)), make_image(1, mask_pixel=(3, 3))
    a.outputs["A"].subtract(b.outputs["A"], combine_masks=False, propagate_std=False)
    assert int(a.get_mask("cr_mask").sum()) == 1
    assert np.allclose(a.get_var("std"), 2.0)


def test_output_binary_op_checks_class_and_shape():
    img = make_image()
    plain = Output(id="P", ext_id=0, ext_slice=(slice(0, 8), slice(0, 6)), data_slice=(slice(0, 8), slice(0, 6)),
                   parent=img)
    with pytest.raises(TypeError, match="same class"):
        img.outputs["A"].subtract(plain)
    other = make_image(1)
    other.outputs["B"].serial_overscan = slice(10, 12)  # narrower image region
    with pytest.raises(ValueError, match="has shape"):
        img.outputs["B"].subtract(other.outputs["B"], region="image")


def test_output_dunders_return_dataarray_and_leave_parent_unchanged():
    a, b = make_image(0), make_image(1)
    before = a.dataset.copy(deep=True)
    out = a.outputs["A"]
    assert isinstance(out + 5, xr.DataArray)
    assert np.allclose(out + 5, before["data"][:, :6] + 5)
    assert np.allclose(5 + out, before["data"][:, :6] + 5)
    assert np.allclose(10 - out, 10 - before["data"][:, :6])
    assert np.allclose(out - b.outputs["A"], before["data"][:, :6].values - b.outputs["A"].data.values)
    assert a.dataset.identical(before)


def test_output_reduce_and_stats_match_do_statistics():
    img = make_image(mask_pixel=(2, 2))
    out = img.outputs["A"]
    imdata, immask = out.get_var(region="image"), out.get_mask("cr_mask", region="image")
    expected = do_statistics(np.ma.masked_array(imdata.values, mask=immask.values), which=STATFUNCS)
    stats = out.stats(region="image", masks="cr_mask")
    assert stats.keys() == expected.keys()
    for key in expected:
        assert np.allclose(stats[key], expected[key])
    assert np.isclose(out.reduce("median", region="image", masks="cr_mask"), expected["median"])
    assert out.reduce("mean", region="image", axis=1).shape == (6,)
    assert np.isclose(out.reduce(np.ma.max, masks=None), out.data.values.max())


def test_output_get_masked_masks_the_pixel():
    out = make_image(mask_pixel=(2, 2)).outputs["A"]
    masked = out.get_masked(masks="cr_mask")
    assert int(masked.mask.sum()) == 1 and masked.shape == out.data.shape


############################################### DetImage ###############################################
def test_detimage_copy_is_independent():
    img = make_image(mask_pixel=(2, 2))
    img.image_type["type"] = "flat"
    new = img.copy()
    assert all(out.parent is new for out in new.outputs.values())
    assert new.id == img.id and new.image_type == img.image_type
    new.outputs["A"].add(1)
    new.outputs["A"].header["X"] = 1
    new.set_var("cr_mask", np.ones(img.shape, dtype=bool))
    new.image_type["type"] = "bias"
    assert not np.allclose(new.data, img.data)
    assert "X" not in img.outputs["A"].header
    assert int(img.get_mask("cr_mask").sum()) == 1
    assert img.image_type["type"] == "flat"


def test_detimage_dunders_return_new_images():
    a, b = make_image(0, mask_pixel=(2, 2)), make_image(1, mask_pixel=(3, 3))
    a_before, b_before = a.dataset.copy(deep=True), b.dataset.copy(deep=True)
    diff = a - b
    total = a + b
    assert diff is not a and total is not a
    assert np.allclose(diff.data, a_before["data"] - b_before["data"])
    assert np.allclose(total.data, a_before["data"] + b_before["data"])
    assert np.allclose(diff.get_var("std"), np.hypot(2.0, 2.0))
    assert int(diff.get_mask("cr_mask").sum()) == 2
    assert a.dataset.identical(a_before) and b.dataset.identical(b_before)
    assert int(a.get_mask("cr_mask").sum()) == 1
    assert np.allclose((a + 1).data, a_before["data"] + 1)
    assert np.allclose((1 - a).data, 1 - a_before["data"])
    assert np.allclose((1 - a).get_var("std"), 2.0)


def test_detimage_inplace_and_named_methods():
    img = make_image()
    before = img.data.copy(deep=True)
    assert img.divide(2, inplace=True) is img
    assert np.allclose(img.data, before / 2)
    assert np.allclose(img.get_var("std"), 1.0)


def test_detimage_binary_op_checks_outputs():
    a = make_image()
    b = DetImage(data=np.zeros(SHAPE))  # default single output "0"
    with pytest.raises(ValueError, match="different outputs"):
        a - b


def test_detimage_integer_data_is_cast_to_float():
    img = DetImage(data=np.arange(12, dtype=np.int32).reshape(3, 4))
    assert np.allclose((img + 0.5).data, np.arange(12).reshape(3, 4) + 0.5)
    assert img.data.dtype == np.int32  # the original is not modified


def test_detimage_reduce_per_output_and_whole():
    img = make_image(mask_pixel=(2, 2))
    medians = img.reduce("median", region="image")
    assert set(medians) == {"A", "B"}
    assert np.isclose(medians["B"], np.median(img.outputs["B"].get_var(region="image")))
    whole = img.reduce("mean", per_output=False, masks="cr_mask")
    assert np.isclose(whole, np.ma.masked_array(img.data.values, mask=img.get_mask("cr_mask").values).mean())
    assert set(img.stats(per_output=False)) >= {"mean", "median", "std"}
    assert np.isclose(img.reduce("mean", per_output=False, region=(slice(0, 4), slice(0, 6)), masks=None),
                      img.data.values[:4, :6].mean())
    with pytest.raises(TypeError, match="tuple or dict of slices"):
        img.reduce(per_output=False, region="image")


def test_detimage_subtract_matches_ptc_diff_loop():
    dp1, dp2 = make_image(0, mask_pixel=(2, 2)), make_image(1, mask_pixel=(3, 3))
    # the per-output loop used in tasks/ptc.py and tasks/tdi.py
    old = dp1.copy()
    for out_id, output in old.outputs.items():
        output.set_var("data", output.data.values - dp2.outputs[out_id].data.values)
    new = dp1.subtract(dp2, combine_masks=False, propagate_std=False)
    assert new.dataset.identical(old.dataset)


def test_detimage_netcdf_roundtrip_keeps_ccdoutputs(tmp_path):
    img = make_image(mask_pixel=(2, 2))
    img.get_mask()
    img.to_netcdf(str(tmp_path / "img"))
    loaded = DetImage.from_netcdf(str(tmp_path / "img.nc"))
    assert all(isinstance(out, CCDOutput) for out in loaded.outputs.values())
    assert np.allclose(loaded.data, img.data)


########################################### FocalPlaneImage ###########################################
def make_focal_plane(seed):
    return FocalPlaneImage(num_detectors=2, det_images=[
        make_image(seed, name="D1", x_cen=6.0, mask_pixel=(2, 2)),
        make_image(seed + 10, name="D2", x_cen=18.0),
    ])


def test_focalplane_ops_rebuild_mosaic():
    fp1, fp2 = make_focal_plane(0), make_focal_plane(1)
    mosaic1, mosaic2 = fp1.data.copy(deep=True), fp2.data.copy(deep=True)
    diff = fp1 - fp2
    assert np.allclose(diff.data, mosaic1 - mosaic2)
    assert np.allclose(fp1.data, mosaic1)  # operands untouched
    assert int(diff.get_mask("cr_mask").sum()) == 1
    assert np.allclose(fp1.multiply(2).data, mosaic1 * 2)
    assert np.allclose((100 - fp1).data, 100 - mosaic1)
    assert fp1.add(fp2, inplace=True) is fp1
    assert np.allclose(fp1.data, mosaic1 + mosaic2)


def test_focalplane_reduce_per_detector():
    fp = make_focal_plane(0)
    medians = fp.reduce("median", region="image")
    assert set(medians) == {"D1", "D2"} and set(medians["D1"]) == {"A", "B"}
    assert set(fp.stats(per_output=False)["D2"]) >= {"mean", "median"}


########################################### operators and checks ###########################################
def test_multiply_divide_and_reflected_operators_propagate_std():
    img = make_image()
    data = img.data.values.copy()
    assert np.allclose((2 * img).data, 2 * data) and np.allclose((img * 2).get_var("std"), 4.0)
    assert np.allclose((img / 4).data, data / 4) and np.allclose((img / 4).get_var("std"), 0.5)
    inverse = 1 / img
    assert np.allclose(inverse.data, 1 / data)
    assert np.allclose(inverse.get_var("std"), 2.0 / data ** 2)
    assert np.allclose((10 - img).data, 10 - data) and np.allclose((10 - img).get_var("std"), 2.0)
    assert np.allclose((img * make_image(1)).data, data * make_image(1).data.values)


def test_output_operators_include_multiply_and_reflected_divide():
    out = make_image().outputs["A"]
    data = out.data.values.copy()
    assert np.allclose(3 * out, 3 * data) and np.allclose(out / 2, data / 2)
    assert np.allclose(2 / out, 2 / data)
    assert np.allclose(out.data, data)  # parent untouched


def test_numpy_operands_on_the_left_defer_to_the_images():
    img = make_image()
    out = img.outputs["A"]
    assert isinstance(np.float64(2) * img, DetImage)
    assert isinstance(np.float64(5) + out, xr.DataArray)
    # an array that does not broadcast to the outputs is an error, not an object array of images
    with pytest.raises(ValueError, match="does not broadcast"):
        np.ones(3) + img


def test_unsupported_operands_raise_type_error():
    img = make_image()
    with pytest.raises(TypeError):
        img + "a"
    with pytest.raises(TypeError):
        img + img.outputs["A"]
    with pytest.raises(TypeError):
        img.outputs["A"] * img


def test_unknown_mask_key_raises():
    img = make_image(mask_pixel=(2, 2))
    with pytest.raises(KeyError, match="has no mask 'crr_mask'"):
        img.outputs["A"].reduce(masks="crr_mask")
    with pytest.raises(KeyError, match="has no mask 'crr_mask'"):
        img.get_masked(masks="crr_mask")
    with pytest.raises(ValueError, match="not a mask name"):
        img.get_mask("cr")
    # 'combined' is fine without masks; a mask set on output A exists, all False, on output B
    assert make_image().get_masked().mask.sum() == 0
    assert img.outputs["B"].stats(masks="cr_mask")["mean"] == img.outputs["B"].stats(masks=None)["mean"]


def test_reduce_rejects_functions_that_ignore_masks():
    out = make_image(mask_pixel=(2, 2)).outputs["A"]
    with pytest.raises(TypeError, match="ignores masked pixels"):
        out.reduce(np.median, masks="cr_mask")
    with pytest.raises(TypeError, match="ignores masked pixels"):
        out.stats({"median": np.median})
    assert np.isclose(out.reduce(np.median, masks=None), np.median(out.data.values))
    assert np.isclose(out.reduce(np.ma.median, masks="cr_mask"), out.reduce("median", masks="cr_mask"))


def test_copy_does_not_share_extra_fields():
    img = make_image()
    img.outputs["A"].notes = ["x"]
    new = img.copy()
    new.outputs["A"].notes.append("y")
    assert img.outputs["A"].notes == ["x"]


def _labelled_image(seed):
    img = make_image(seed)
    labels = xr.zeros_like(img.data, dtype=np.int32)
    labels[2, 3] = 1
    img.set_var("cosmic_ray_label", labels)
    return img


def test_copy_keeps_independent_labels():
    img = _labelled_image(0)
    new = img.copy()
    assert new.get_var("cosmic_ray_label").identical(img.get_var("cosmic_ray_label"))
    new.set_var("cosmic_ray_label", np.full(img.shape, 7, dtype=np.int32))
    assert int(img.get_var("cosmic_ray_label").max()) == 1


def test_arithmetic_keeps_the_left_operands_labels(tmp_path):
    a, b = _labelled_image(0), make_image(1)
    for result in (a + 1, a - b, 2 * a):
        assert result.get_var("cosmic_ray_label").identical(a.get_var("cosmic_ray_label"))
    assert (b - a).label_names == []
    diff = a - b
    diff.to_netcdf(tmp_path / "diff")
    assert DetImage.from_netcdf(tmp_path / "diff.nc").get_var("cosmic_ray_label").equals(a.get_var("cosmic_ray_label"))

