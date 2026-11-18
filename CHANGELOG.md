# Changelog

## Unreleased

### Added
- Arithmetic on detector images: `Output`, `DetImage` and `FocalPlaneImage` have `add`, `subtract`, `multiply`,
  `divide` (and the general `unary_op` / `binary_op`), and the operators `+ - * /` including reflected forms such as
  `1 - image` and `2 / image`. Operations can be limited to a named region of an output ('full', 'image',
  'serial_overscan', ...), OR the masks of the two operands and propagate a 'std' data variable (operands are
  treated as independent). `DetImage`/`FocalPlaneImage` operations return a new object unless `inplace=True`;
  `Output.add`/`subtract`/... change the parent image in place, while `Output` operators return an `xr.DataArray`.
- `Output.reduce` / `Output.stats` (and on `DetImage`, `FocalPlaneImage`): masked statistics of a named region, per
  output or per detector.
- `DetImage.copy()` (a cheaper replacement for `copy.deepcopy`; copies labels too), `DetImage.get_masked_array()`,
  `Output.get_region()`.
- `core`: `arithmetic`, `propagate_std`, `as_masked_array`, `reduce_array`, `check_mask_aware`.

### Changed
- `datamodels/image.py` is split into `meta.py`, `output.py`, `image.py` and `bundle.py`; imports from
  `eregion.datamodels` are unchanged.
- `reduce`/`stats` raise `TypeError` for functions that ignore masks (e.g. `np.median`, scipy.stats functions);
  use a statistic name or a `numpy.ma` function. They, and `DetImage.get_masked_array`, raise `KeyError` for a
  mask key that is neither 'combined' nor an existing mask, instead of silently ignoring it.
- Tasks use the new methods (bias, scan and difference-image subtraction, masked statistics, image copies), with
  unchanged results except that: integer pixel data is converted to float before arithmetic (float results written
  back into integer data were truncated); `BiasSubtraction` requires the master bias to have the same output ids
  and output class as the image; the masked statistics of PTC, MasterDark and TDI raise `KeyError` for an unknown
  mask key.

### Fixed
- `DetImage.to_netcdf` accepts `pathlib.Path` and appends `.nc` instead of replacing the part of the name after
  the last dot; `ImageBundle.save` works again.
- `FocalPlaneImage` keeps detector masks that changed after the detector was added.
- `Output.get_masked_output_data` no longer fails when the output has masks.
