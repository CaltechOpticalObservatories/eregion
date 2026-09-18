from astropy.io import fits
import numpy as np
from string import Template
import pytest

from eregion.tasks.imagegen import ImageCreator

_CONFIG_YAML_TPL = """
detector_type: CCD
detector_output_class: CCDOutput

objects:
    - name: "det"
      class: DetImage
      header_index: 0
      filename_format: "*.fits"
      properties:
        x_size: 100
        y_size: 200
        pixel_size: 0.1
      focal_plane_position:
        x_cen: 0
        y_cen: 0
      outputs:
      - id: "One"
        ext_id: ${First}
        ext_slice: [!slice [0, 100], !slice [0, 100]]
        data_slice: [!slice [0,100], !slice [0, 100]]
      - id: "Two"
        ext_id: ${Second}
        ext_slice: [!slice [0, 100], !slice [0, 100]]
        data_slice: [!slice [100, 200], !slice [0, 100]]
"""


@pytest.fixture(scope="module")
def example_FITS(tmp_path_factory):
    prihdu = fits.PrimaryHDU()

    hdu1 = fits.CompImageHDU(
        header=fits.Header({"EXTNAME": "badabing"}),
        data=np.ones(dtype=np.uint8, shape=(100, 100)) * 12,
    )

    hdu2 = fits.CompImageHDU(
        header=fits.Header({"EXTNAME": "badaboom"}),
        data=np.ones(dtype=np.uint8, shape=(100, 100)) * 15,
    )

    hdul = fits.HDUList([prihdu, hdu1, hdu2])

    temp_dir = tmp_path_factory.mktemp("FITSdata")
    fpath = temp_dir / "file.fits"
    hdul.writeto(str(fpath))
    print(fpath)
    return fpath


def _setup_crtr(fitspath, first, second):
    tpl = Template(_CONFIG_YAML_TPL)
    yaml = tpl.substitute(First=first, Second=second)
    crtr = ImageCreator(detector_config=yaml, n_jobs=1)

    return crtr


def _check_outputs(result, valmap):
    # should only be one image, multiple outputs tho
    assert len(result.data) == 1
    for opname, opobj in result.data[0].outputs.items():
        arr = opobj.data
        assert np.all(arr == valmap[opname])


def test_ImageCreator_load_with_indices(example_FITS):
    crtr = _setup_crtr(example_FITS, 1, 2)

    # NOTE: input_source doesn't work if it's a PosixPath or iterable of PosixPaths.
    # That will annoy me enough to fix one day
    result = crtr.run(input_source=str(example_FITS))

    valmap = {"One": 12, "Two": 15}
    _check_outputs(result, valmap)


def test_ImageCreator_load_with_indices_out_of_order(example_FITS):
    crtr = _setup_crtr(example_FITS, 2, 1)

    result = crtr.run(input_source=str(example_FITS))
    valmap = {"One": 15, "Two": 12}
    _check_outputs(result, valmap)


def test_ImageCreator_with_names(example_FITS):
    crtr = _setup_crtr(example_FITS, "badaboom", "badabing")
    result = crtr.run(input_source=str(example_FITS))

    valmap = {"One": 15, "Two": 12}

    _check_outputs(result, valmap)


def test_extname_in_header(example_FITS):
    crtr = _setup_crtr(example_FITS, "badaboom", "badabing")
    result = crtr.run(input_source=str(example_FITS))

    assert result.data[0].outputs["One"].header["EXTNAME"] == "badaboom"
    assert result.data[0].outputs["Two"].header["EXTNAME"] == "badabing"
