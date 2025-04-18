from itertools import combinations

import pandas as pd
from shapely.geometry import box

from agents.common.models.assets import AssetComparisonElement
from agents.common.models.shapes import LabeledBox
from agents.common.stats.outlier.base_outlier_stats import (
    BaseOutlierStats,
    ComparisonOutlierGroupItem,
)


class InterClassOutlierStats(BaseOutlierStats):
    def __init__(
        self,
        df: pd.DataFrame,
        cooccurrence_matrix: pd.DataFrame,
        overlap_matrix: pd.DataFrame,
        co_threshold_low_ratio: float = 0.01,
        co_threshold_ratio: float = 0.06,
        overlap_threshold_ratio: float = 0.06,
    ):
        super().__init__(df)
        self.co_matrix = cooccurrence_matrix
        self.overlap_matrix = overlap_matrix
        self.co_threshold_low_ratio = co_threshold_low_ratio
        self.co_threshold_ratio = co_threshold_ratio
        self.overlap_threshold_ratio = overlap_threshold_ratio

    def _prepare_data(self) -> None:
        pass

    def _compute_outliers(self) -> None:
        self.outlier_groups["unexpected_cooccurrence"] = {
            "unexpected_pairs": self._unexpected_cooccurrence()
        }
        self.outlier_groups["unexpected_overlap"] = {
            "unexpected_overlap": self._unexpected_overlap()
        }
        self.outlier_groups["missing_expected_cooccurrence"] = {
            "missing_expected": self._missing_expected_cooccurrence()
        }

    def _to_labeled_box(self, row: pd.Series) -> LabeledBox:
        return LabeledBox(
            x=row["x"],
            y=row["y"],
            w=row["w"],
            h=row["h"],
            label_id=row["label_id"],
        )

    def _unexpected_cooccurrence(self) -> ComparisonOutlierGroupItem:
        total_images = self.df["asset_id"].nunique()
        min_count = max(2, total_images * self.co_threshold_low_ratio)
        min_number_of_assets = 3

        rare_pairs = {
            frozenset((a, b))
            for a in self.co_matrix.columns
            for b in self.co_matrix.columns
            if a < b and self.co_matrix.at[a, b] <= min_count
        }

        pair_map: dict[tuple[str, str], list[AssetComparisonElement]] = {}

        for asset_id, group in self.df.groupby("asset_id"):
            labels = set(group["label"])
            for pair in combinations(labels, 2):
                key = frozenset(pair)
                if key in rare_pairs:
                    sorted_pair = tuple(sorted(pair))
                    affected = group[group["label"].isin(pair)]
                    element = AssetComparisonElement(
                        id=str(asset_id),
                        actual=[
                            self._to_labeled_box(r) for _, r in affected.iterrows()
                        ],
                        expected=[],
                    )
                    pair_map.setdefault(sorted_pair, []).append(element)

        filtered_pairs = {
            (label_1, label_2): elements
            for (label_1, label_2), elements in pair_map.items()
            if len(elements) >= min_number_of_assets
        }

        return {
            "description": (
                "Pairs of labels that rarely appear together in the dataset"
            ),
            "pairs": [
                {"label_1": l1, "label_2": l2, "elements": elems}
                for (l1, l2), elems in filtered_pairs.items()
            ],
        }

    def _missing_expected_cooccurrence(self) -> ComparisonOutlierGroupItem:
        total_images = self.df["asset_id"].nunique()
        min_expected = max(3, total_images * self.co_threshold_ratio)
        min_number_of_assets = 3

        expected_pairs = {
            frozenset((a, b))
            for a in self.co_matrix.columns
            for b in self.co_matrix.columns
            if a < b and self.co_matrix.at[a, b] >= min_expected
        }

        pair_map: dict[tuple[str, str], list[AssetComparisonElement]] = {}

        for asset_id, group in self.df.groupby("asset_id"):
            labels = set(group["label"])
            for pair in expected_pairs:
                if len(pair & labels) == 1:
                    sorted_pair = tuple(sorted(pair))
                    affected = group[group["label"].isin(pair)]
                    element = AssetComparisonElement(
                        id=str(asset_id),
                        actual=[
                            self._to_labeled_box(r) for _, r in affected.iterrows()
                        ],
                        expected=[],
                    )
                    pair_map.setdefault(sorted_pair, []).append(element)

        filtered_pairs = {
            (label_1, label_2): elements
            for (label_1, label_2), elements in pair_map.items()
            if len(elements) >= min_number_of_assets
        }

        return {
            "description": (
                "Pairs of labels that usually appear together, but are missing in certain images"
            ),
            "pairs": [
                {
                    "label_1": label_1,
                    "label_2": label_2,
                    "elements": elements,
                }
                for (label_1, label_2), elements in filtered_pairs.items()
            ],
        }

    def _unexpected_overlap(self) -> ComparisonOutlierGroupItem:
        total_images = self.df["asset_id"].nunique()
        min_overlap_count = max(3, total_images * self.overlap_threshold_ratio)
        min_number_of_assets = 3

        low_overlap_pairs = {
            frozenset((a, b))
            for a in self.overlap_matrix.columns
            for b in self.overlap_matrix.columns
            if a < b and self.overlap_matrix.at[a, b] <= min_overlap_count
        }

        pair_map: dict[tuple[str, str], list[AssetComparisonElement]] = {}

        for asset_id, group in self.df.groupby("asset_id"):
            boxes = []
            for _, row in group.iterrows():
                try:
                    b = box(
                        row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"]
                    )
                    boxes.append((row["label"], b, row["shape_id"], row))
                except Exception:
                    continue

            for (label_a, box_a, _, row_a), (label_b, box_b, _, row_b) in combinations(
                boxes, 2
            ):
                key = frozenset((label_a, label_b))
                if key in low_overlap_pairs:
                    iou = box_a.intersection(box_b).area / box_a.union(box_b).area
                    if iou > 0.1:
                        sorted_pair = tuple(sorted([label_a, label_b]))
                        df_pair = pd.DataFrame([row_a, row_b])
                        element = AssetComparisonElement(
                            id=str(asset_id),
                            actual=[
                                self._to_labeled_box(r) for _, r in df_pair.iterrows()
                            ],
                            expected=[],
                        )
                        pair_map.setdefault(sorted_pair, []).append(element)

        filtered_pairs = {
            (label_1, label_2): elements
            for (label_1, label_2), elements in pair_map.items()
            if len(elements) >= min_number_of_assets
        }

        return {
            "description": (
                "Label pairs that almost never overlap in the dataset but were found with significant spatial intersection"
            ),
            "pairs": [
                {
                    "label_1": label_1,
                    "label_2": label_2,
                    "elements": elements,
                }
                for (label_1, label_2), elements in filtered_pairs.items()
            ],
        }
