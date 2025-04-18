from typing import Literal

from pydantic import BaseModel


class Shapes(BaseModel):
    type: Literal["shape-list"]
    ids: list[str]


class LabeledBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    label_id: str


# === Label-only comparison ===
class LabelComparisonElement(BaseModel):
    id: str
    actual: str
    expected: str


# === Bounding box + label comparison ===
class ShapeLabelComparisonElement(BaseModel):
    id: str
    actual: LabeledBox
    expected: LabeledBox


# === Global container ===
class ShapeComparison(BaseModel):
    type: Literal["shape-comparisons"]
    mode: Literal["label", "shape-label"]
    data: list[LabelComparisonElement | ShapeLabelComparisonElement]
