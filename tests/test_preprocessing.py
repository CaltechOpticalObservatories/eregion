import numpy as np
import pytest
import xarray as xr
from astropy.io.fits.card import Undefined

from eregion.datamodels import DetImage, CCDOutput
from eregion.tasks.preprocessing import VignetteMasking, CosmicRayMasking, ScanSubtraction


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


def _ccd_with_cosmics(with_std=False):
    ny, nx = 120, 130
    rng = np.random.default_rng(2)
    arr = rng.normal(0, 5, (ny, nx))
    arr[10:110, 10:120] += rng.normal(1000.0, np.sqrt(1000.0), (100, 110))  # illuminated image region
    crs = {"image": (50, 50), "overscan": (60, 125), "premasked": (80, 80)}
    for y, x in crs.values():
        arr[y, x] += 5000.0
    coords = {"y": np.arange(ny), "x": np.arange(nx)}
    data = xr.DataArray(arr, dims=("y", "x"), coords=coords)
    if with_std:
        std = np.full((ny, nx), 5.0)
        std[10:110, 10:120] = np.sqrt(1000.0 + 25.0)
        data = xr.Dataset({"data": data, "std": xr.DataArray(std, dims=("y", "x"), coords=coords)})
    ccd = CCDOutput(
        id="C",
        ext_id=0,
        ext_slice=(slice(0, ny), slice(0, nx)),
        data_slice=(slice(0, ny), slice(0, nx)),
        serial_prescan=slice(0, 10),
        serial_overscan=slice(120, 130),
        parallel_prescan=slice(0, 10),
        parallel_overscan=slice(110, 120),
        parallel_axis="y",
    )
    det = DetImage(data=data, output_objects={"C": ccd})
    det.image_type["bias_subtracted"] = True
    existing = xr.zeros_like(det.outputs["C"].data).astype(bool)
    existing[crs["premasked"]] = True
    det.outputs["C"].masks = existing.to_dataset(name="vignette_mask")
    return det, crs


@pytest.mark.parametrize("with_std, lacosmic_args", [
    (False, {"effective_gain": 1.0, "readnoise": 5.0}),  # gain/readnoise from args
    (False, None),  # gain=1, readnoise estimated from serial overscan
    (True, None),  # std plane
])
def test_cosmic_ray_masking_ccd_searches_regions_and_respects_existing_masks(with_std, lacosmic_args):
    det, crs = _ccd_with_cosmics(with_std)
    out = CosmicRayMasking(lacosmic_args=lacosmic_args)._process_single_image(det)
    mask = out.outputs["C"].masks["cosmic_ray_mask"].values

    assert mask.shape == (120, 130)
    assert mask[crs["image"]]
    assert mask[crs["overscan"]]
    assert not mask[crs["premasked"]]
    assert mask.sum() < 50
    assert out.image_type["cosmic_ray_masked"]


def test_cosmic_ray_masking_simple_flags_image_region_edge():
    det, crs = _ccd_with_cosmics()
    out = CosmicRayMasking(method="simple")._process_single_image(det)
    mask = out.outputs["C"].masks["cosmic_ray_mask"].values
    assert mask[crs["image"]]
    # step between overscan and illuminated image region is picked up when searching the whole output
    assert mask[10:110, 10].sum() > 50


def test_cosmic_ray_masking_estimates_readnoise_from_overscan():
    det, _ = _ccd_with_cosmics()
    noise_args = CosmicRayMasking()._noise_args(det.outputs["C"], has_std=False)
    assert noise_args["effective_gain"] == 1.0
    assert noise_args["readnoise"] == pytest.approx(5.0, rel=0.1)


def test_cosmic_ray_masking_rejects_per_output_args():
    with pytest.raises(ValueError, match="error"):
        CosmicRayMasking(lacosmic_args={"error": np.ones((2, 2))})


def test_cosmic_ray_masking_labels_cosmic_rays_when_enabled():
    det, crs = _ccd_with_cosmics()
    out = CosmicRayMasking(label=True)._process_single_image(det)
    mask = out.outputs["C"].masks["cosmic_ray_mask"].values
    labels = out.labels["cosmic_ray_labels"]

    assert np.array_equal(labels.values > 0, mask)
    assert labels.attrs["num_labels"] == len(np.unique(labels.values[labels.values > 0]))
    assert labels.values[crs["image"]] > 0 and labels.values[crs["overscan"]] > 0
    assert labels.values[crs["image"]] != labels.values[crs["overscan"]]


def test_cosmic_ray_masking_does_not_label_by_default():
    det, _ = _ccd_with_cosmics()
    assert CosmicRayMasking()._process_single_image(det).labels is None


def test_cosmic_ray_masking_rejects_bad_label_connectivity():
    with pytest.raises(ValueError, match="label_connectivity"):
        CosmicRayMasking(label_connectivity=4)


def _ccd_with_overscan(serial_overscan=slice(120, 130), bias=500.0, bad_pixel=None, bad_value=np.nan):
    """Build a DetImage with a 10-pixel-wide serial overscan at a known bias level."""
    ny, nx = 120, 130
    rng = np.random.default_rng(7)
    arr = rng.normal(bias, 2.0, (ny, nx))
    arr[10:110, 10:120] += 1000.0  # illuminated image region
    if bad_pixel is not None:
        arr[bad_pixel] = bad_value
    data = xr.DataArray(arr, dims=("y", "x"), coords={"y": np.arange(ny), "x": np.arange(nx)})
    ccd = CCDOutput(
        id="C",
        ext_id=0,
        ext_slice=(slice(0, ny), slice(0, nx)),
        data_slice=(slice(0, ny), slice(0, nx)),
        serial_prescan=slice(0, 10),
        serial_overscan=serial_overscan,
        parallel_prescan=slice(0, 10),
        parallel_overscan=slice(110, 120),
        parallel_axis="y",
    )
    return DetImage(data=data, output_objects={"C": ccd})


def test_scan_subtraction_removes_the_overscan_level_and_records_its_statistics():
    out = ScanSubtraction("serial_overscan")._process_single_image(_ccd_with_overscan())
    data = out.outputs["C"].data.values
    hdr = out.outputs["C"].header

    # every row is offset by the median of its own serial overscan, so those medians are now zero
    assert np.median(data[:, 120:130], axis=1) == pytest.approx(0.0, abs=1e-9)
    assert np.median(data[10:110, 10:120]) == pytest.approx(1000.0, abs=2.0)
    assert hdr["serial_overscan_median"] == pytest.approx(500.0, abs=1.0)
    assert hdr["serial_overscan_mean"] == pytest.approx(500.0, abs=1.0)
    assert hdr["serial_overscan_std"] == pytest.approx(2.0, rel=0.3)
    assert out.image_type["serial_overscan_subtracted"]


def test_scan_subtraction_trims_the_scan_region_before_measuring_it():
    det = _ccd_with_overscan()
    det.get_data("data").values[:, 120] = 1e6  # bad leading overscan column, trimmed away
    out = ScanSubtraction("serial_overscan", trim_start=1)._process_single_image(det)
    assert out.outputs["C"].header["serial_overscan_median"] == pytest.approx(500.0, abs=1.0)


def test_scan_subtraction_rejects_an_empty_scan_region():
    det = _ccd_with_overscan(serial_overscan=slice(130, 130))
    with pytest.raises(ValueError, match="holds no pixels"):
        ScanSubtraction("serial_overscan")._process_single_image(det)


@pytest.mark.parametrize("trim_start, trim_end", [
    (5, 5),    # exactly nothing left
    (10, 0),
    (0, 10),
    (0, 15),   # over-trim: the stop index would go negative and silently wrap
    (-1, 0),   # a negative trim would also wrap
])
def test_scan_subtraction_rejects_trims_that_empty_the_scan_region(trim_start, trim_end):
    task = ScanSubtraction("serial_overscan", trim_start=trim_start, trim_end=trim_end)
    with pytest.raises(ValueError, match="leave no pixels"):
        task._process_single_image(_ccd_with_overscan())  # 10-pixel serial overscan


def test_scan_subtraction_accepts_a_trim_leaving_a_single_pixel():
    out = ScanSubtraction("serial_overscan", trim_start=4, trim_end=5)._process_single_image(_ccd_with_overscan())
    assert np.isfinite(out.outputs["C"].header["serial_overscan_median"])


def test_scan_subtraction_writes_undefined_cards_for_nan_scan_statistics():
    det = _ccd_with_overscan(bad_pixel=(3, 125), bad_value=np.nan)
    out = ScanSubtraction("serial_overscan")._process_single_image(det)
    hdr = out.outputs["C"].header

    for stat in ("mean", "median", "std", "mad"):
        key = f"serial_overscan_{stat}"
        assert key in hdr
        assert isinstance(hdr.cards[key].value, Undefined)
        assert hdr[key] is None
    assert hdr.tostring()  # the header is still writable; this used to raise on the NaN card

    # row 3's overscan median is NaN, so only that row is NaN after subtraction
    data = out.outputs["C"].data.values
    assert np.isnan(data[3]).all()
    assert np.isfinite(data[np.arange(data.shape[0]) != 3]).all()


def test_scan_subtraction_keeps_the_finite_statistics_when_only_some_are_non_finite():
    det = _ccd_with_overscan(bad_pixel=(3, 125), bad_value=np.inf)
    out = ScanSubtraction("serial_overscan")._process_single_image(det)
    hdr = out.outputs["C"].header

    # one +inf among 1200 overscan pixels poisons mean and std, but not the median or the MAD
    assert hdr["serial_overscan_mean"] is None
    assert hdr["serial_overscan_std"] is None
    assert hdr["serial_overscan_median"] == pytest.approx(500.0, abs=1.0)
    assert hdr["serial_overscan_mad"] == pytest.approx(2.0, rel=0.3)
    assert hdr.tostring()
