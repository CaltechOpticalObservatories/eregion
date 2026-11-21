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
- `DetImage.copy()`, a cheaper replacement for `copy.deepcopy`.
- `core`: `arithmetic`, `propagate_std`, `as_masked_array`, `reduce_array`, `check_mask_aware`.
- `utils`: `to_slicer` and `to_label_slicer` (pure; they replace `decrease_slicer_stop_index`, which changed its
  argument in place).

### Changed
- **One Dataset per image.** A `DetImage` keeps its data, masks and labels in one `xr.Dataset` (`img.dataset`), named
  by kind: `<prefix>_mask` vars are bool masks, `<prefix>_label` vars integer labels (0 is background), anything
  else is data. Masks cover the whole image, so a mask set on one output exists (all False) on the others.
  `'combined'`, `'all'`, `'mask'` and `'label'` are reserved names. `FocalPlaneImage` mosaics every kind the same
  way. The cosmic ray labels are now `cosmic_ray_label`.
- **One get/set API on DetImage, Output and FocalPlaneImage.** `region` is a pixel slicer, or on Output a region name
  ('image', 'serial_overscan', ...). `masks` is `'combined'`, a mask name, a list of names, or `None`.

  | Old | New |
  |---|---|
  | `img.get_data(name)`, `img.labels[name]`, `img.masks[name]` | `img.get_var(name, region=None)` |
  | `img.get_data('all')` | `img.dataset` (all vars), `img.data_names` / `mask_names` / `label_names` |
  | `img.get_mask(key)`, `img.combined_mask` | `img.get_mask(masks='combined', region=None)`, always a bool array |
  | `img.get_masked_array(...)`, `img.get_masked_data(...)` | `img.get_masked(name, masks, region, mask_invalid)` |
  | `img.set_data(array)`, `img.set_data_slice(...)`, `img.set_labels(...)` | `img.set_var(name, values, region=None)` |
  | `img.set_data(dataset)` | `img.set_vars(mapping_or_dataset)` |
  | `img.masks = None` | `img.remove_vars(*img.mask_names)` |
  | `DetImage(data=loader)`, `img.set_data(loader)` | `DetImage(loader=loader)`, `img.set_loader(loader)`, `img.load()` |
  | `output.get_output_data`, `get_image_region`, `get_region`, `get_prescan`, `get_overscan`, `get_scan` | `output.get_var(name, region='image' / 'serial_overscan' / ...)` |
  | `output.get_scan(..., corner=True)` | `output.get_var(region=output.region_slicer('serial_overscan', corner=True))` |
  | `output.image_region` | `output.region_slicer('image')` |
  | `output.set_mask(key, mask)` | `output.set_var('<prefix>_mask', mask, combine=True)` |
  | `output.set_data_in_parent(values, slicer)` | `output.set_var('data', values, region=slicer)` |
  | `output.get_output_mask`, `output.combined_mask`, `output.masks` | `output.get_mask(masks)`, `output.mask_names` |
  | `reduce`/`stats`/`show(data_var=, mask_key=, use_mask=, with_mask=)` | `name=`, `masks=` (`None` for no masking) |

- `set_var` checks names, dtypes, shapes and coordinates, and raises instead of NaN-filling or truncating.
  `get_var` may return a view of the image; write through `set_var`.
- `unload()` drops the data vars only (masks and labels stay). For an image without a loader it raises unless
  called with `discard=True`; SuperCombine uses that to unload every input, as before.
- netcdf files carry `eregion_format: 2`. Files from older versions load with their masks cast to bool and renamed
  `<name>_mask`, and `cosmic_ray_labels` renamed `cosmic_ray_label`.
- The `*_masked` image_type flags of the masking tasks report whether that task's mask exists.
- `datamodels/image.py` is split into `meta.py`, `output.py`, `image.py` and `bundle.py`; imports from
  `eregion.datamodels` are unchanged.
- `reduce`/`stats` raise `TypeError` for functions that ignore masks (e.g. `np.median`, scipy.stats functions);
  use a statistic name or a `numpy.ma` function. Unknown mask names raise `KeyError` instead of being ignored.
- Tasks use the new methods (bias, scan and difference-image subtraction, masked statistics, image copies), with
  unchanged results except that: integer pixel data is converted to float before arithmetic (float results written
  back into integer data were truncated); `BiasSubtraction` requires the master bias to have the same output ids
  and output class as the image; the masked statistics of PTC, MasterDark and TDI raise `KeyError` for an unknown
  mask key.

### Fixed
- `DetImage.to_netcdf` accepts `pathlib.Path` and appends `.nc` instead of replacing the part of the name after
  the last dot; `ImageBundle.save` works again.
- `FocalPlaneImage` keeps detector masks that changed after the detector was added.
- `MasterBias`/`MasterDark` no longer turn a pixel masked in some frames into NaN: masked pixels are left out of the
  combination, and `count`/`std` reflect that. A master carries one mask, 'combine_mask' (pixels masked in every
  frame), instead of the first frame's masks, and no longer inherits the first frame's labels.
- `MasterDark.run(mask_key=...)` now uses `mask_key` for the combination; the master's stats use its 'combine_mask'.
- `TDIExtractPTC` no longer crashes on masked images; `n_masked` is a per-row count like the other statistics.
- `LinBin` no longer raises `KeyError` with the default `mask_key`, bins every mask along with the data, and uses
  `mask_key` for its statistics.
- `ImageBundle.load` restores the saved order (image_2 before image_10).
