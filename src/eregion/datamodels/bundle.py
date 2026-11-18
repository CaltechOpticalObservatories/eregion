from typing import Generic, TypeVar, Self
import pathlib
import pandas as pd
from copy import deepcopy

from eregion.datamodels.image import DetImage

TImage = TypeVar("TImage")


class ImageBundle(Generic[TImage]):
    """
    Class to hold a list of images. Contains methods to tabulate metadata of the images for easy filtering.

    Attributes
    ----------
    images: list[TImage]
        List of input images.
    list: pd.DataFrame
        DataFrame containing image identifying metadata from image_class.image_type.

    Methods
    -------
    __call__(pd_query: str = '')
        Returns a new ImageBundle with images filtered based on the provided pandas query string. Calls filter() internally.
    filter(pd_query: str = '')
        Returns a list of images filtered based on the provided pandas query string.
    save(filepath: str, **kwargs)
        Saves all images in the ImageBundle to the specified folder path using their to_netcdf() method.
    load(filepath: str)
        Loads all netcdf files from the specified folder path and creates an ImageBundle from them using the from_netcdf() method of the image_class.
    __iter__()
        Returns an iterator over the list of images in the ImageBundle.
    __getitem__(i)
        Returns the image at index i in the ImageBundle.
    __setitem__(key, value)
        Sets the image at index key to value in the ImageBundle. Validates that value is of the correct image_class type.
    __add__(other: Self)
        Combines two ImageBundle instances of the same type and returns a new ImageBundle containing images from both.
    append(image)
        Appends a new image to the ImageBundle after validating its type.
    extend(other: Self)
        Extends the ImageBundle with images from another ImageBundle of the same type.

    """

    image_class: type[TImage] = DetImage

    def __init__(self, images: TImage | list[TImage] | None = None):
        """
        :param images: image_class | list[image_class] | None
            image_class could be DetImage, FocalPlaneImage, etc.
        :return: ImageBundle instance.
        """
        images = (
            images
            if isinstance(images, list)
            else [images] if images is not None else []
        )
        self.images: list[TImage] = [self._validate_image(image) for image in images]
        self.list: pd.DataFrame | None = None
        self._tabulate()

    def _validate_image(self, image: TImage) -> TImage:
        if not isinstance(image, self.image_class):
            raise TypeError(
                f"Expected {self.image_class.__name__}, got {type(image).__name__}"
            )
        return image

    def _tabulate(self):
        """
        Loops through list of images and creates a dataframe from their image_type dict.

        Default columns are `det_id`, `filename`, `object`, containing image_class.id, image_class.meta['filename']
        and the image_class object itself. Rest of the columns/values are keys/values in the image_class.image_type.
        """
        tab = []
        for i, image in enumerate(self.images):
            imtype = deepcopy(image.image_type)
            imtype["det_id"] = image.id
            imtype["filename"] = image.meta.get("filename", None)
            imtype["object"] = image
            tab.append(imtype)
        self.list = pd.DataFrame(tab)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        """
        Create an ImageBundle from a pandas DataFrame. The DataFrame must contain a column named 'object' with image_class instances.
        :param df: pd.DataFrame
            DataFrame containing image_class instances in a column named 'object'.
        :return: ImageBundle instance.
        """
        if "object" not in df.columns:
            raise ValueError(
                "DataFrame must contain a column named 'object' with image_class instances."
            )
        images = df["object"].to_list()
        return cls(images=images)

    def __call__(self, pd_query: str = ""):
        """
        Returns a new ImageBundle with images filtered based on the provided pandas query string.
        :param pd_query: str
            Query string to filter the dataframe. Should be a valid pandas query string.
        :return: ImageBundle
            A new ImageBundle with the filtered images.
        """
        return type(self)(self.filter(pd_query))

    def filter(self, pd_query: str = "") -> list[TImage]:
        """
        Filters images based on the provided pandas query string. Returns a list of filtered images.
        :param pd_query: str
            Query string to filter the dataframe. Should be a valid pandas query string.
        :return: List of filtered DetImages.
        """
        df = self.list.query(pd_query) if pd_query != "" else self.list
        return df["object"].to_list()

    def groupby(self, by, sort=False, dropna=False, **kwargs):
        """
        Wrapper for pandas groupby on self.list dataframe. Returns a pandas groupby object.
        :param by: pandas groupby **by** parameter.
        :param sort: pandas groupby **sort** parameter. Default is False for faster operation.
        :param dropna: pandas groupby **dropna** parameter. Default is False to keep NaN groups.
        :param kwargs: pandas groupby additional keyword arguments.
        :return: pandas groupby object.
        """
        missing_keys = set(by).difference(self.list.columns)
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise KeyError(f"Groupby keys not found in DataFrame columns: {missing}")
        return self.list.groupby(by, sort=sort, dropna=dropna, **kwargs)

    def save(self, filepath: str, **kwargs):
        """
        Call to_netcdf() for each image object, and save them in one folder path
        :param filepath: str
            Path to folder where images will be saved. The images are saved with image_{i}.nc filenames where
            "i" is their index in the ImageBundle list.
        :param **kwargs: Additional keyword arguments to pass to the to_netcdf() method of the image_class.
        """
        # check if image_class has to_netcdf method
        if not hasattr(self.image_class, "to_netcdf"):
            raise AttributeError(
                f"{self.image_class.__name__} does not have a to_netcdf() method."
            )

        filepath = pathlib.Path(filepath).resolve()
        filepath.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(self.images):
            image.to_netcdf(filepath / f"image_{i}.nc", **kwargs)

    @classmethod
    def load(cls, filepath: str):
        """
        Load all the netcdf files from {filepath} folder and create an ImageBundle from them.
        :param filepath: str
            Path to folder where images are saved. Looks for filenames of format image_{i}.nc
        :return: ImageBundle instance
        """
        # check if image_class has from_netcdf method
        if not hasattr(cls.image_class, "from_netcdf"):
            raise AttributeError(
                f"{cls.image_class.__name__} does not have a from_netcdf() method."
            )

        files = sorted(pathlib.Path(filepath).resolve().glob("*.nc"))
        images = []
        for file in files:
            im = cls.image_class.from_netcdf(file)
            images.append(im)
        return cls(images=images)

    def __repr__(self):
        return f"ImageBundle with {len(self)} images, {repr(self.list)}"

    def _repr_html_(self):
        return self.list._repr_html_()

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    def __getitem__(self, i):
        return self.images[i]

    def __setitem__(self, key, value):
        self.images[key] = self._validate_image(value)
        self._tabulate()

    def __add__(self, other: Self):
        if type(self) is not type(other):
            raise TypeError(
                f"Cannot combine {self.__class__.__name__} with {other.__class__.__name__}."
            )
        return type(self)(self.images + other.images)

    def append(self, image):
        self.images.append(self._validate_image(image))
        self._tabulate()

    def extend(self, other: Self):
        if type(self) is not type(other):
            raise TypeError(
                f"Cannot extend {self.__class__.__name__} with {other.__class__.__name__}."
            )
        self.images.extend(other.images)
        self._tabulate()
