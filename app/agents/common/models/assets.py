from typing import Literal

from pydantic import BaseModel

from agents.common.models.shapes import LabeledBox


class Assets(BaseModel):
    type: Literal["asset-list"]
    ids: list[str]


class AssetComparisonElement(BaseModel):
    id: str
    actual: list[LabeledBox]
    expected: list[LabeledBox]


class AssetComparison(BaseModel):
    type: Literal["asset-comparisons"]
    data: list[AssetComparisonElement]
