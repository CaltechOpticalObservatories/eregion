from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


@dataclass
class Output:
    id: int
    filename: str
    ext_id: int
    ext_slice: tuple[slice, slice] # Slice to extract from the FITS extension ext_id
    data_slice: tuple[slice, slice] # Slice where the data of this output will go in the DetImage full data array
    serial_prescan: slice = slice(0, 0)
    serial_overscan: slice = slice(0, 0)
    parallel_prescan: slice = slice(0, 0)
    parallel_overscan: slice = slice(0, 0)
    parallel_axis: int = 0 # Axis along which parallel readout occurs, by default 0 (i.e., rows)
    readout_pixel: tuple[int, int] = (0, 0) # Pixel coordinates of the readout amplifier
    fits_header: dict = None
    meta: dict = None
    serial_axis: int = int(not bool(parallel_axis)) # Axis along which serial readout occurs, inferred from parallel_axis
    parent = None

    def show(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(6, 6), tight_layout=True)
        image = self._data
        im = ax.imshow(image, **kwargs)
        ## Shade the prescan and overscan regions
        spandict = {0: ax.axvspan, 1: ax.axhspan}

        scan_types = [
            ("serial_prescan", self.serial_prescan, self.parallel_axis, "gold", "Serial Prescan"),
            ("serial_overscan", self.serial_overscan, self.parallel_axis, "red", "Serial Overscan"),
            ("parallel_prescan", self.parallel_prescan, self.serial_axis, "cyan", "Parallel Prescan"),
            ("parallel_overscan", self.parallel_overscan, self.serial_axis, "blue", "Parallel Overscan"),
        ]

        for name, scan, axis, color, label in scan_types:
            # Determine start index
            if scan.start is not None:
                scan_start = (image.shape[axis] + scan.start) if scan.start < 0 else scan.start
            else:
                scan_start = 0
            # Determine stop index
            if scan.stop is not None:
                scan_stop = (image.shape[axis] + scan.stop) if scan.stop < 0 else scan.stop
            else:
                scan_stop = image.shape[axis]
            # Draw the region
            spandict[axis](scan_start, scan_stop, color=color, alpha=0.3, label=label)

        plt.colorbar(im, ax=ax)
        plt.legend(loc=(0.01,1.01), fontsize=8)
        plt.show()
        return ax

    @property
    def _data(self) -> np.ndarray:
        # Parent's data if this Output was attached to a DetImage (`output.parent = det_image`).
        if hasattr(self, "parent") and getattr(self.parent, "data", None) is not None:
            data = self.parent.data
        else:
            raise ValueError(
                "No data available: attach this Output to a DetImage (`output.parent = det`) or set `output.data`.")
        return slice_data(data, self.data_slice)

    @property
    def _prescan(self, type: str) -> np.ndarray:
        if type == "serial":
            slc = self.serial_prescan
            axis = self.serial_axis
        elif type == "parallel":
            slc = self.parallel_prescan
            axis = self.parallel_axis
        else:
            raise ValueError(f"Unknown prescan type: {type}")
        data = self._data
        if axis == 0:
            return data[slc, :]
        else:
            return data[:, slc]

    @property
    def _overscan(self, type: str) -> np.ndarray:
        if type == "serial":
            slc = self.serial_overscan
            axis = self.serial_axis
        elif type == "parallel":
            slc = self.parallel_overscan
            axis = self.parallel_axis
        else:
            raise ValueError(f"Unknown overscan type: {type}")
        data = self._data
        if axis == 0:
            return data[slc, :]
        else:
            return data[:, slc]

def slice_data(data: np.ndarray, slicer: tuple[slice, slice]) -> np.ndarray:
    return data[slicer]

# Base class for detector image.
class DetImage:
    required_keys = []

    def __init__(self, data: np.ndarray = None,
                 output_objects: list[Output] = None,
                 image_type: str = None,
                 **kwargs):
        self.data = data
        self.image_type = image_type
        self.meta = {}
        if kwargs is not None:
            self.validate_kwargs(kwargs)
        self.meta.update(kwargs)
        self.focal_plane = None

        if output_objects is None and data is not None:
            # If no output objects are provided, create a default one covering the whole data
            logger.info("Creating default output object for DetImage, since none were provided but data is given.")
            default_output = Output(id=0, filename=self.meta["filename"] if "filename" in self.meta else "unknown",
                ext_id=1, data_slice=(slice(0, data.shape[0]), slice(0, data.shape[1])), parent=self)
            self.outputs = [default_output]
        else:
            self.outputs = output_objects or []
        for output in self.outputs:
            output.parent = self

    def validate_kwargs(self, kwargs):
        for key in self.required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required keyword argument: {key}")

    def _output_by_id(self, output_id: int) -> Output:
        for output in self.outputs:
            if output.id == output_id:
                return output
        raise ValueError(f"Output with id {output_id} not found.")

    def _output_by_index(self, index: int) -> Output:
        if index < 0 or index >= len(self.outputs):
            raise IndexError(f"Output index {index} out of range.")
        return self.outputs[index]

    def add_output(self, output: Output):
        output.parent = self
        self.outputs.append(output)

    @property
    def num_outputs(self):
        return len(self.outputs)

    def show(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(6, 6), tight_layout=True)
        im = ax.imshow(self.data, **kwargs)
        plt.colorbar(im, ax=ax)
        plt.show()
        return ax

class FocalPlaneImage:
    def __init__(self, num_detectors: int, dim: tuple[int, int], det_images: list[DetImage] = None, **kwargs):
        self.meta = {}
        if kwargs is not None:
            self.meta.update(kwargs)
        self.num_detectors = num_detectors
        self.det_images = []
        for det_image in det_images:
            self.add_DetImage(det_image)
        self.dim = dim
        self.data = np.empty(self.dim)
        self.construct_focal_plane_image()

    def validate_det_image(self, det_image: DetImage):
        required_keys = ["properties", "focal_plane_position"]
        required_subkeys = {"properties": ["pixel_size", "x_size", "y_size"], "focal_plane_position": ["x_cen", "y_cen"]}
        for key in required_keys:
            if key not in det_image.meta.keys():
                raise ValueError(f"Each DetImage must have '{key}' in its meta to be added to FocalPlaneImage.")
            else:
                for subkey in required_subkeys[key]:
                    if subkey not in det_image.meta[key].keys():
                        raise ValueError(f"Each DetImage's '{key}' meta must include '{subkey}'.")


    def construct_focal_plane_image(self):
        # Construct a combined focal plane image from the detector images, using focal_plane_position in meta of det_images
        # Read each det_image's center position and calculate the corners
        frames = []
        pixsize = self.det_images[0].meta["properties"]["pixel_size"]
        for det_image in self.det_images:
            pos = det_image.meta["focal_plane_position"]
            xhalf = (det_image.meta["properties"]["x_size"] / 2)
            yhalf = (det_image.meta["properties"]["y_size"] / 2)
            # Calculate corners in pixel units, with origin at the center of the focal plane
            corners = {
                "det_id": det_image.meta["name"],
                "x_min": pos["x_cen"] / pixsize - xhalf,
                "x_max": pos["x_cen"] / pixsize + xhalf,
                "y_min": pos["y_cen"] / pixsize - yhalf,
                "y_max": pos["y_cen"] / pixsize + yhalf
            }
            frames.append(corners)
        frames_df = pd.DataFrame(frames)
        # Verify that there are no overlapping detectors, i.e. area covered inside corners should not overlap
        for i, row1 in frames_df.iterrows():
            for j, row2 in frames_df.iterrows():
                if i >= j:
                    continue
                if not (row1["x_max"] <= row2["x_min"] or row1["x_min"] >= row2["x_max"] or
                        row1["y_max"] <= row2["y_min"] or row1["y_min"] >= row2["y_max"]):
                    raise ValueError(f"Detectors {row1['det_id']} and {row2['det_id']} overlap in focal plane.")
        ## If no overlaps, stack the images into a larger array
        # Verify the size of the focal plane image
        fp_ymin = int(frames_df["y_min"].min())
        fp_ymax = int(frames_df["y_max"].max())
        fp_xmin = int(frames_df["x_min"].min())
        fp_xmax = int(frames_df["x_max"].max())
        # Create empty focal plane array
        if (fp_ymax - fp_ymin != self.dim[0]) or (fp_xmax - fp_xmin != self.dim[1]):
            # Raise warning if the provided dim does not match calculated dim
            logger.warning(f"Provided focal plane dim {self.dim} does not match calculated dim {(fp_ymax - fp_ymin, fp_xmax - fp_xmin)} from det_images' positions.")

        # Calculate the positions to place each det_image in the focal plane array
        frames_df["y_min_fp"] = -1*(frames_df["y_max"] - self.dim[0]//2) # Flip y-axis
        frames_df["y_max_fp"] = -1*(frames_df["y_min"] - self.dim[0]//2)
        frames_df["x_min_fp"] = frames_df["x_min"] + self.dim[1]//2
        frames_df["x_max_fp"] = frames_df["x_max"] + self.dim[1]//2
        # Place each det_image's data into the focal plane array
        for i, det_image in enumerate(self.det_images):
            yslc = slice(int(frames_df.loc[i, "y_min_fp"]), int(frames_df.loc[i, "y_max_fp"]))
            xslc = slice(int(frames_df.loc[i, "x_min_fp"]), int(frames_df.loc[i, "x_max_fp"]))
            self.data[yslc, xslc] = det_image.data
        self.frames_df = frames_df

    def add_DetImage(self, det_image: DetImage):
        if len(self.det_images)==self.num_detectors:
            raise ValueError(f"Number of det_images ({len(self.det_images)}) is already at the limit of number of detectors in this focal plane ({self.num_detectors}).")
        self.validate_det_image(det_image)
        det_image.focal_plane = self
        self.det_images.append(det_image)

    def show(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(8, 8), tight_layout=True)
        im = ax.imshow(self.data, **kwargs)
        # Draw detector boundaries
        for i, row in self.frames_df.iterrows():
            rect = plt.Rectangle((row["x_min_fp"], row["y_min_fp"]),
                                 row["x_max_fp"] - row["x_min_fp"],
                                 row["y_max_fp"] - row["y_min_fp"],
                                 linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(row["x_min_fp"] +150, row["y_min_fp"] +150, row["det_id"], color='white', fontsize=8)
        plt.colorbar(im, ax=ax)
        plt.show()
        return ax

