# Changelog

## Unreleased

### Changed
- `datamodels/image.py` is split into `meta.py`, `output.py`, `image.py` and `bundle.py`; imports from
  `eregion.datamodels` are unchanged.

### Fixed
- `DetImage.to_netcdf` accepts `pathlib.Path` and appends `.nc` instead of replacing the part of the name after
  the last dot; `ImageBundle.save` works again.
