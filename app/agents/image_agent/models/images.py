from enum import Enum

from pydantic import BaseModel, ConfigDict

from agents.common.models.actions import PossibleActions


class ImageIssueType(str, Enum):
    BLURRY = "BLURRY"
    CORRUPTED = "CORRUPTED"
    LOW_CONTRAST = "LOW-CONTRAST"
    LOW_LUMINANCE = "LOW-LUMINANCE"
    HIGH_CONTRAST = "HIGH-CONTRAST"
    HIGH_LUMINANCE = "HIGH-LUMINANCE"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"
    OUTLIER = "OUTLIER"


class Image(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    asset_id: str


class QualityImage(BaseModel):
    asset_id: str
    issues: list[ImageIssueType]
    potential_actions: list[PossibleActions]


class ImageGroup(BaseModel):
    group_type: ImageIssueType
    potential_actions: list[PossibleActions]
    images: list[Image]
