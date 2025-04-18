from enum import Enum

from pydantic import BaseModel, ConfigDict


class ShapeIssueType(str, Enum):
    AMBIGIOUS_CLASS = "AMBIGIOUS_CLASS"
    WRONG_CLASS = "WRONG_CLASS"
    BOX_TIGHTNESS = "BOX_TIGHTNESS"
    SMALL_AREA = "SMALL_AREA"
    OUTSIDE_OF_IMAGE = "OUTSIDE_OF_IMAGE"


class Shape(BaseModel):
    id: str
    x: int
    y: int
    w: int
    h: int
    issue: ShapeIssueType
    label_annotated: str
    label_suggested: str


class Annotation(BaseModel):
    id: str
    shapes: list[Shape]


class Image(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    asset_id: str
    annotation: Annotation


class TightnessShape(BaseModel):
    id: str
    issue: ShapeIssueType
    box: list[int]  # (x,y,w,h)
    suggestion: list[int]  # (x,y,w,h)
    iou: float
    label: str
    label_id: str


class TightnessAnnotation(BaseModel):
    id: str
    shapes: list[TightnessShape]


class TightnessImage(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    asset_id: str
    annotation: TightnessAnnotation
