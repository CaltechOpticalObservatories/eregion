import numpy as np
import xarray as xr

from eregion.datamodels import ImageBundle
from eregion.tasks.calibration import MasterBias, MasterDark
from tests.image_helpers import make_image


def make_frames(kind, masked):
    """
    Three frames of the 2-output test image. masked maps a mask key to the frames (index -> list of (y, x))
    where that key masks a pixel of output A.
    """
    frames = []
    for i in range(3):
        img = make_image(i)
        img.meta.filename = f"{kind}{i}.fits"
        img.image_type.update({"type": kind, "exptime": 0.0})
        out = img.outputs["A"]
        for key, per_frame in masked.items():
            mask = xr.zeros_like(out.data, dtype=bool)
            for y, x in per_frame.get(i, []):
                mask.loc[{"y": y, "x": x}] = True
            out.set_var(key, mask, combine=True)
        frames.append(img)
    return frames


def test_masterbias_leaves_masked_pixels_out_of_the_combination():
    # (2, 2) masked in frame 0 only, (3, 3) masked in every frame
    frames = make_frames("bias", {"cr_mask": {0: [(2, 2), (3, 3)], 1: [(3, 3)], 2: [(3, 3)]}})
    master = MasterBias(groupby_keys=["det_id", "exptime"]).run(images=frames).master_bias[0]

    others = np.array([float(f.data.sel(y=2, x=2)) for f in frames[1:]])
    assert np.isclose(float(master.data.sel(y=2, x=2)), np.median(others))
    assert int(master.get_var("count").sel(y=2, x=2)) == 2
    assert np.isclose(float(master.get_var("std").sel(y=2, x=2)), np.std(others))

    assert np.isnan(float(master.data.sel(y=3, x=3)))
    assert int(master.get_var("count").sel(y=3, x=3)) == 0

    unmasked = np.array([float(f.data.sel(y=4, x=4)) for f in frames])
    assert np.isclose(float(master.data.sel(y=4, x=4)), np.median(unmasked))
    assert int(master.get_var("count").sel(y=4, x=4)) == 3


def test_master_carries_only_the_combine_mask():
    frames = make_frames("bias", {"cr_mask": {0: [(2, 2), (3, 3)], 1: [(3, 3)], 2: [(3, 3)]}})
    master = MasterBias(groupby_keys=["det_id", "exptime"]).run(images=frames).master_bias[0]
    assert master.mask_names == ["combine_mask"]
    assert bool(master.get_mask("combine_mask").sel(y=3, x=3))
    assert int(master.get_mask("combine_mask").sum()) == 1
    # every output has the mask, all-False where nothing was masked in every frame
    assert int(master.outputs["B"].get_mask("combine_mask").sum()) == 0
    # the input frames keep their own masks
    assert int(frames[0].get_mask("cr_mask").sum()) == 2


def test_master_does_not_inherit_the_first_frames_labels():
    frames = make_frames("bias", {})
    labels = xr.zeros_like(frames[0].data, dtype=np.int32)
    labels[2, 3] = 1
    frames[0].set_var("cosmic_ray_label", labels)
    master = MasterBias(groupby_keys=["det_id", "exptime"]).run(images=frames).master_bias[0]
    assert master.label_names == []
    assert frames[0].label_names == ["cosmic_ray_label"]


def test_masterdark_combines_with_the_requested_mask_key(caplog):
    # 'cr_mask' masks (2, 2) in frame 0; 'other_mask' masks (4, 4) in frame 0 and must be ignored
    frames = make_frames("dark", {"cr_mask": {0: [(2, 2)]}, "other_mask": {0: [(4, 4)]}})
    res = MasterDark(groupby_keys=["det_id", "exptime"]).run(images=frames, mask_key="cr_mask")
    master = res.master_dark[0]
    assert int(master.get_var("count").sel(y=2, x=2)) == 2
    assert int(master.get_var("count").sel(y=4, x=4)) == 3
    # the master's stats use its combine_mask, which has no pixels here, without looking for 'cr_mask' on the master
    assert set(res.stats["n_masked"]) == {0}
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_masterdark_stats_count_pixels_masked_in_every_frame():
    frames = make_frames("dark", {"cr_mask": {0: [(3, 3)], 1: [(3, 3)], 2: [(3, 3)]}})
    res = MasterDark(groupby_keys=["det_id", "exptime"]).run(images=frames, mask_key="cr_mask")
    n_masked = dict(zip(res.stats["output"], res.stats["n_masked"]))
    assert n_masked == {"A": 1, "B": 0}


def test_mastercombine_without_masks_is_a_plain_median():
    frames = make_frames("bias", {})
    master = MasterBias(groupby_keys=["det_id", "exptime"]).run(images=ImageBundle(frames)).master_bias[0]
    assert np.allclose(master.data, np.median([f.data.values for f in frames], axis=0))
    assert int(master.get_var("count").min()) == 3
