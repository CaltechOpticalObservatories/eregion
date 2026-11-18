from typing import Optional, Any
from .mappable import Mappable
from pydantic import Field, ConfigDict

############################################### META CLASSES FOR DETIMAGE #############################################
class DetectorProperties(Mappable):
    """
    Physical and sampling properties for a detector tile.
    """

    pixel_size: float = Field(gt=0, description="Pixel size in mm.")
    x_size: int = Field(gt=0, description="Tile width in pixels.")
    y_size: int = Field(gt=0, description="Tile height in pixels.")

    model_config = ConfigDict(extra="allow")


class FocalPlanePosition(Mappable):
    """
    Center position of the detector on the focal plane (same units as pixel_size, typically mm).
    """

    x_cen: float = Field(..., description="Center X in length units.")
    y_cen: float = Field(..., description="Center Y in length units.")

    model_config = ConfigDict(extra="allow")


class DetImageMeta(Mappable):
    """
    Metadata for a detector image, validated for focal-plane assembly.
    """

    name: str = Field(default=None)
    filename: Optional[str] = Field(default=None)
    properties: DetectorProperties
    focal_plane_position: Optional[FocalPlanePosition]
    image_type: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of identifying keys for this image, e.g. type, exptime, mode, etc.",
    )

    model_config = ConfigDict(extra="allow")