import json

import numpy as np
import xarray as xr
from astropy.io import fits

from eregion.datamodels import (
    DetectorProperties,
    FocalPlanePosition,
    DetImageMeta,
    Output,
    CCDOutput,
    DetImage,
    FocalPlaneImage,
    ImageBundle,
    FPImageBundle,
)


def make_dataarray(shape=(10, 10)):
    # Create a white noise DataArray for testing
    arr = np.random.rand(*shape)
    return xr.DataArray(arr, dims=("y", "x"), coords={"y": np.arange(shape[0]), "x": np.arange(shape[1])})


def test_detector_properties_fields():
    props = DetectorProperties(pixel_size=0.01, x_size=1024, y_size=2048)
    assert props.pixel_size == 0.01 and props.x_size == 1024 and props.y_size == 2048


def test_focal_plane_position_fields():
    pos = FocalPlanePosition(x_cen=100.0, y_cen=50.0)
    assert pos.x_cen == 100.0 and pos.y_cen == 50.0


def test_detimage_meta_validation_dict_to_model():
    props = {"pixel_size":0.01, "x_size":10, "y_size":12}
    pos = {"x_cen":0.0, "y_cen":0.0}
    meta_dict = {"name": "D1", "properties": props, "focal_plane_position": pos}
    meta = DetImageMeta.model_validate(meta_dict)
    assert (meta.name == "D1" and isinstance(meta.properties, DetectorProperties)
            and isinstance(meta.focal_plane_position, FocalPlanePosition))


def test_mappable_update_recursively_merges_models_and_dicts():
    meta = DetImageMeta(
        name="D1",
        properties=DetectorProperties(pixel_size=0.01, x_size=10, y_size=12),
        focal_plane_position=FocalPlanePosition(x_cen=0.0, y_cen=0.0),
        image_type={"type": "flat", "exptime": 10},
    )

    meta.update(
        {
            "properties": {"pixel_size": 0.02},
            "image_type": {"filter": "r"},
            "name": "D1-updated",
        }
    )

    assert meta.name == "D1-updated"
    assert meta.properties == DetectorProperties(pixel_size=0.02, x_size=10, y_size=12)
    assert meta.image_type == {"type": "flat", "exptime": 10, "filter": "r"}


def test_output_data_slices_parent_data():
    data = make_dataarray((10, 12))
    det = DetImage(data=data)
    # Replace default full output with a smaller one
    out = Output(
        id="A",
        ext_id=1,
        ext_slice=(slice(0, 5), slice(0, 6)),
        data_slice=(slice(2, 7), slice(3, 9)),
        parent=det,
    )
    det.outputs = {"A": out}
    sub = out.data
    assert sub.shape == (5, 6)
    assert np.all(sub.values == data.values[2:7, 3:9])


def test_output_json_combines_base_and_field_serializers():
    output = Output(
        id="A",
        ext_id=0,
        ext_slice=(slice(0, 2), slice(0, 2)),
        data_slice=(slice(0, 2), slice(0, 2)),
        header=fits.Header({"TEST": 1}),
    )

    serialized = json.loads(output.to_json(exclude={"masks", "parent"}))

    assert serialized["header"] == {"TEST": 1}
    assert serialized["input_slice"][0]["__eregion_json_type__"] == "slice"


def test_lazy_detimage_netcdf_serializes_output_header(tmp_path):
    def load_data(_: str):
        return [np.ones((2, 2))], [fits.Header({"TEST": 1})]

    output = Output(
        id="A",
        ext_id=0,
        ext_slice=(slice(0, 2), slice(0, 2)),
        data_slice=(slice(0, 2), slice(0, 2)),
    )
    image = DetImage(
        data=load_data,
        output_objects={"A": output},
        meta=DetImageMeta(
            name="D1",
            filename="synthetic.fits",
            properties=DetectorProperties(pixel_size=0.01, x_size=2, y_size=2),
            focal_plane_position=None,
        ),
    )

    path = tmp_path / "lazy-image.nc"
    image.to_netcdf(str(path))

    assert image.outputs["A"].header["TEST"] == 1
    assert not hasattr(image.outputs["A"], "fits_header")
    saved_outputs = json.loads(xr.load_dataset(path).attrs["outputs"])
    assert json.loads(saved_outputs["A"])["header"] == {"TEST": 1}


def test_ccdoutput_serial_axis_and_regions():
    data = make_dataarray((8, 10))
    det = DetImage(data=data)
    ccd = CCDOutput(
        id="C",
        ext_id=1,
        ext_slice=(slice(0, 8), slice(0, 10)),
        data_slice=(slice(0, 8), slice(0, 10)),
        parent=det,
        serial_prescan=slice(0, 2),
        serial_overscan=slice(8, 10),  # out of bounds handled by xarray indexing
        parallel_prescan=slice(0, 2),
        parallel_overscan=slice(6, 8),
        parallel_axis="y",
    )
    # serial axis should be x if parallel is y
    assert ccd.serial_axis == "x"
    sp = ccd.get_prescan("serial", corner=False)
    pp = ccd.get_prescan("parallel", corner=False)
    sp1 = ccd.get_prescan("serial", corner=True)
    pp1 = ccd.get_prescan("parallel", corner=True)
    assert sp.shape[1] == 2 and pp.shape[0] == 2
    assert sp.shape[0] == 4 and pp.shape[1] == 6
    assert sp1.shape[0] == 8 and pp1.shape[1] == 10


def test_detimage_default_output_creation_and_lookup():
    data = make_dataarray((7, 9))
    det = DetImage(data=data)
    # default output exists
    assert det.num_outputs == 1
    out = det.outputs["0"]
    assert out.parent is det
    # add another output
    new_out = Output(
        id="1",
        ext_id=0,
        ext_slice=(slice(0, 3), slice(0, 4)),
        data_slice=(slice(1, 4), slice(2, 6)),
    )
    det.add_output(new_out)
    assert det.num_outputs == 2 and det.outputs["1"].parent is det


def test_focalplaneimage_construct_and_place_tiles():
    # Create two non-overlapping tiles
    props = DetectorProperties(pixel_size=1.0, x_size=4, y_size=4)
    img1 = DetImage(
        data=make_dataarray((4, 4)),
        meta=DetImageMeta(
            name="D1",
            properties=props,
            focal_plane_position=FocalPlanePosition(x_cen=2.0, y_cen=2.0),
        ),
    )
    img2 = DetImage(
        data=make_dataarray((4, 4)) + 100,
        meta=DetImageMeta(
            name="D2",
            properties=props,
            focal_plane_position=FocalPlanePosition(x_cen=8.0, y_cen=2.0),
        ),
    )
    fp = FocalPlaneImage(num_detectors=2, dim=(4, 10), det_images=[img1, img2])
    assert fp.data.shape == (4, 10)
    # Ensure both tiles are placed
    assert np.count_nonzero(fp.data.values) > 0

def test_imagebundle_construct_and_access():
    props = DetectorProperties(pixel_size=1.0, x_size=4, y_size=4)
    imgs = []
    for i in range(4):
        for imtype in [{'type': 'bias', 'exptime': 0}, {'type': 'dark', 'exptime': 10}, {'type': 'flat', 'exptime': 10}]:
            img = DetImage(
                data=make_dataarray((4, 4)),
                properties=props,
                focal_plane_position=FocalPlanePosition(x_cen=0.0, y_cen=0.0),
                image_type=imtype,
                name=f"D{i}")
            imgs.append(img)

    bundle = ImageBundle(images=imgs)
    assert len(bundle) == 12
    # Test filtering by type
    assert len(bundle('det_id == "D0"')) == 3
    assert len(bundle('type == "bias"')) == 4
    assert len(bundle('exptime == 10')) == 8
    assert len(bundle('exptime == 10 & type == "flat"')) == 4
    assert len(bundle('exptime == 10 & type == "dark" & det_id == "D1"')) == 1
    assert [isinstance(im, DetImage) for im in bundle()]


def test_imagebundle_combination_is_pure_and_type_checked():
    props = DetectorProperties(pixel_size=1.0, x_size=4, y_size=4)
    first_image = DetImage(
        data=make_dataarray((4, 4)),
        properties=props,
        focal_plane_position=FocalPlanePosition(x_cen=0.0, y_cen=0.0),
        image_type={"type": "bias"},
        name="D1",
    )
    second_image = DetImage(
        data=make_dataarray((4, 4)),
        properties=props,
        focal_plane_position=FocalPlanePosition(x_cen=0.0, y_cen=0.0),
        image_type={"type": "bias"},
        name="D2",
    )
    first = ImageBundle(first_image)
    second = ImageBundle(second_image)

    combined = first + second

    assert len(first) == 1
    assert len(second) == 1
    assert len(combined) == 2

    with np.testing.assert_raises(TypeError):
        first[0] = FocalPlaneImage(num_detectors=1)
    first.extend(second)
    assert len(first) == 2


def test_fpimagebundle_filter_preserves_subtype():
    det_image = DetImage(data=make_dataarray((4, 4)),
                         meta=DetImageMeta(properties={'x_size': 4, 'y_size': 4, 'pixel_size': 1.0},
                                           focal_plane_position={'x_cen': 0.0, 'y_cen': 0.0}))
    focal_plane = FocalPlaneImage(num_detectors=1, det_images=[det_image])
    bundle = FPImageBundle(images=focal_plane)

    filtered = bundle()

    assert isinstance(filtered, FPImageBundle)
    assert filtered[0] is focal_plane
    with np.testing.assert_raises(TypeError):
        _ = bundle + ImageBundle()