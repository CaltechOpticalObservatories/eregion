from pathlib import Path
import shutil

import numpy as np
import pytest
from astropy.io import fits

from eregion.utils.io_utils import (
    guess_image_type_from_header,
    is_archive_file,
    is_directory,
    is_fits_file,
    load_image_fits,
    parse_list_of_files,
    search_directory_for_fits_files,
)


def _write_simple_fits(path: Path):
    primary = fits.PrimaryHDU(data=np.arange(4).reshape(2, 2))
    ext = fits.ImageHDU(data=np.ones((2, 2)))
    fits.HDUList([primary, ext]).writeto(path)


def test_is_fits_file_and_missing_path(tmp_path):
    fits_path = tmp_path / "a.fits"
    txt_path = tmp_path / "a.txt"
    _write_simple_fits(fits_path)
    txt_path.write_text("x")

    assert is_fits_file(str(fits_path)) is True
    assert is_fits_file(str(txt_path)) is False

    with pytest.raises(FileNotFoundError):
        is_fits_file(str(tmp_path / "missing.fits"))


def test_is_archive_file_recognizes_zip_and_excludes_fits_gz(tmp_path):
    plain = tmp_path / "a.txt"
    plain.write_text("data")
    archive = shutil.make_archive(str(tmp_path / "bundle"), "zip", root_dir=tmp_path, base_dir="a.txt")

    fits_gz = tmp_path / "image.fits.gz"
    fits_gz.write_text("not real fits, extension only")

    assert is_archive_file(archive) is True
    assert is_archive_file(str(fits_gz)) is False


def test_is_directory(tmp_path):
    file_path = tmp_path / "f.txt"
    file_path.write_text("x")

    assert is_directory(str(tmp_path)) is True
    assert is_directory(str(file_path)) is False


def test_search_directory_for_fits_files_is_recursive_and_sorted(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    a = tmp_path / "b.fits"
    b = sub / "a.fits.fz"
    _write_simple_fits(a)
    b.write_text("compressed-placeholder")

    found = search_directory_for_fits_files(str(tmp_path))
    assert found == sorted([str(a), str(b)])


def test_parse_list_of_files_unpacks_archive_and_returns_fits_files(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    fits_path = src / "in_archive.fits"
    _write_simple_fits(fits_path)
    archive = shutil.make_archive(str(tmp_path / "bundle"), "zip", root_dir=src, base_dir=".")

    out = parse_list_of_files([archive])

    assert len(out) == 1
    assert out[0].endswith("in_archive.fits")


def test_load_image_fits_reads_hdus_and_handles_none(tmp_path):
    fits_path = tmp_path / "img.fits"
    _write_simple_fits(fits_path)

    data, headers = load_image_fits(str(fits_path))
    assert len(data) == 2
    assert len(headers) == 2
    assert data[0].shape == (2, 2)

    empty_data, empty_headers = load_image_fits(None)
    assert empty_data == []
    assert empty_headers == []


def test_guess_image_type_from_header_defaults_and_custom_keywords():
    headers = [{"OBJECT": "BIAS", "EXPTIME": "10"}]
    out = guess_image_type_from_header(headers)
    assert out == {"type": "bias", "exptime": "10"}

    custom = guess_image_type_from_header(
        [{"MYTYPE": "FLAT", "MYEXP": "20"}],
        keywords={"type": ["MYTYPE"], "exptime": ["MYEXP"]},
    )
    assert custom == {"type": "flat", "exptime": "20"}
