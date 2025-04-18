from abc import ABC, abstractmethod
from typing import Self, TypedDict, TypeVar

import pandas as pd

from agents.common.models.assets import AssetComparisonElement

T = TypeVar("T", bound="BaseOutlierStats")


class ShapeOutlierElement(TypedDict):
    label: str
    ids: list[str]


class ShapeOutlierGroupItem(TypedDict):
    description: str
    elements: list[ShapeOutlierElement]


class ComparisonPairEntry(TypedDict):
    label_1: str
    label_2: str
    elements: list[AssetComparisonElement]


class ComparisonOutlierGroupItem(TypedDict):
    description: str
    pairs: list[ComparisonPairEntry]


class AssetOutlierGroupItem(TypedDict):
    description: str
    ids: list[str]


class BaseOutlierStats(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.outlier_groups: dict[
            str,
            dict[
                str,
                AssetOutlierGroupItem
                | ComparisonOutlierGroupItem
                | ShapeOutlierGroupItem,
            ],
        ] = {}
        # Structure attendue pour outlier_groups:
        # {
        #     "luminance": {
        #         "high_outliers": {
        #             "description": "Images with high luminance",
        #             "ids": ["136962", "425964"]
        #         },
        #         "normal": {
        #             "description": "Typical luminance values",
        #             "ids": ["136hh962", "42596ff4"]
        #         }
        #     },
        #     ...
        # }

    @abstractmethod
    def _prepare_data(self) -> None:
        pass

    @abstractmethod
    def _compute_outliers(self) -> None:
        pass

    def compute(self) -> Self:
        self._prepare_data()
        self._compute_outliers()
        return self
