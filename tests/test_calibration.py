import numpy as np
import pytest

from eregion.datamodels import DetImage, ImageBundle
from eregion.tasks.calibration import MasterCombine


@pytest.mark.parametrize("method, npfunc", [("median", np.median), ("mean", np.mean)])
def test_master_combine_stores_combined_image_in_data(method, npfunc):
    rng = np.random.default_rng(0)
    stack = rng.normal(100.0, 5.0, size=(5, 4, 6))
    images = [DetImage(data=arr, meta={"name": "D1", "filename": f"bias{i}.fits", "image_type": {"type": "bias"}})
              for i, arr in enumerate(stack)]

    master = MasterCombine(method=method).run(ImageBundle(images)).data[0]

    planes = master.get_data("all")
    assert set(planes.data_vars) == {"data", "std", "count"}
    np.testing.assert_allclose(planes["data"].values, npfunc(stack, axis=0))
    np.testing.assert_array_equal(planes["count"].values, len(stack))
