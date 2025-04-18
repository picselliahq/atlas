from typing import Any

import pandas as pd
from shapely.geometry import box

from agents.common.charts.types import BarChart, ChartType
from agents.common.stats.analysis.base_chart_stats import BaseChartStats


class InterClassStats(BaseChartStats["InterClassStats"]):
    def __init__(
        self,
        df: pd.DataFrame,
        iou_threshold_mean: float = 0.0,
        iou_threshold_count: float = 0.2,
    ):
        super().__init__(df=df)
        self.iou_threshold_mean = iou_threshold_mean
        self.iou_threshold_count = iou_threshold_count
        self.stats: dict[str, Any] = {}

    def _prepare_data(self):
        self.df["area"] = self.df["w"] * self.df["h"]
        self.df["image_area"] = self.df["image_width"] * self.df["image_height"]
        self.df["relative_area"] = self.df["area"] / self.df["image_area"]
        self.df["aspect_ratio"] = self.df["w"] / self.df["h"]
        self.df["w_norm"] = self.df["w"] / self.df["image_width"]
        self.df["h_norm"] = self.df["h"] / self.df["image_height"]
        self.df["area_norm"] = self.df["area"] / self.df["image_area"]

    def _compute_stats(self):
        self._compute_cooccurrence()
        self._compute_mean_iou_per_class_pair()
        self._compute_overlap_count_matrix()
        self._compute_summary_stats()

    def _group_boxes_by_asset(self):
        for asset_id, group in self.df.groupby("asset_id"):
            boxes = []
            for _, row in group.iterrows():
                try:
                    geom = box(
                        row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"]
                    )
                    boxes.append((row["label"], geom))
                except KeyError:
                    continue
            yield asset_id, boxes

    def _compute_cooccurrence(self):
        cooccurrence = (
            self.df.groupby(["asset_id", "label"])
            .size()
            .unstack(fill_value=0)
            .astype(bool)
            .astype(int)
        )
        matrix = cooccurrence.T @ cooccurrence
        self.stats["cooccurrence_matrix"] = matrix
        self.stats["cooccurrence_dict"] = matrix.to_dict()

    def _compute_mean_iou_per_class_pair(self):
        from collections import defaultdict
        from itertools import combinations

        pairwise_ious: defaultdict[tuple[str, str], list[float]] = defaultdict(list)

        for _, boxes in self._group_boxes_by_asset():
            for (label_a, box_a), (label_b, box_b) in combinations(boxes, 2):
                iou = box_a.intersection(box_b).area / box_a.union(box_b).area
                if iou >= self.iou_threshold_mean:
                    pair = tuple(sorted((label_a, label_b)))
                    pairwise_ious[pair].append(iou)

        iou_matrix: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for (label_a, label_b), values in pairwise_ious.items():
            mean_iou = sum(values) / len(values)
            iou_matrix[label_a][label_b] = mean_iou
            iou_matrix[label_b][label_a] = mean_iou

        matrix_df = pd.DataFrame(iou_matrix).fillna(0.0)
        self.stats["mean_iou_per_pair_matrix"] = matrix_df
        self.stats["iou_threshold_mean"] = self.iou_threshold_mean

    def _compute_overlap_count_matrix(self):
        from collections import defaultdict
        from itertools import combinations

        overlap_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        for _, boxes in self._group_boxes_by_asset():
            for (label_a, box_a), (label_b, box_b) in combinations(boxes, 2):
                iou = box_a.intersection(box_b).area / box_a.union(box_b).area
                if iou >= self.iou_threshold_count:
                    pair = tuple(sorted((label_a, label_b)))
                    overlap_counts[pair] += 1

        labels = sorted(set(self.df["label"]))
        count_matrix = pd.DataFrame(0, index=labels, columns=labels)

        for (label_a, label_b), count in overlap_counts.items():
            count_matrix.loc[label_a, label_b] = count
            count_matrix.loc[label_b, label_a] = count

        self.stats["overlap_count_matrix"] = count_matrix
        self.stats["iou_threshold_count"] = self.iou_threshold_count

    def _compute_summary_stats(self):
        self._compute_top_cooccurrences()
        self._compute_top_overlaps()
        self._compute_unusual_classes()

    def _get_top_pairs(
        self, matrix: pd.DataFrame, top_n: int = 10
    ) -> list[tuple[str, str, int]]:
        labels = matrix.index.tolist()
        result = []

        for i in range(len(labels)):
            for j in range(i, len(labels)):
                a, b = labels[i], labels[j]
                value = matrix.at[a, b]
                result.append((a, b, int(value)))

        result.sort(key=lambda x: x[2], reverse=True)
        return result[:top_n]

    def _compute_top_cooccurrences(self, top_n: int = 10):
        matrix = self.stats["cooccurrence_matrix"]
        self.stats["top_cooccurrences"] = self._get_top_pairs(matrix, top_n)

    def _compute_top_overlaps(self, top_n: int = 10):
        matrix = self.stats["overlap_count_matrix"]
        self.stats["top_overlaps"] = self._get_top_pairs(matrix, top_n)

    def _compute_unusual_classes(self):
        matrix = self.stats["cooccurrence_matrix"]

        total_co = matrix.sum(axis=1)

        never_co = total_co[total_co == 0].index.tolist()

        threshold = total_co.mean() + total_co.std()
        frequent_co = (
            total_co[total_co > threshold].sort_values(ascending=False).index.tolist()
        )

        self.stats["classes_with_no_cooccurrence"] = never_co
        self.stats["classes_with_high_cooccurrence"] = frequent_co

    def _generate_all_chart_groups(self):
        self.chart_groups = {
            "cooccurrence": {
                "top_cooccurrences": self._chart_top_cooccurrences(),
            },
            "overlap": {
                "top_overlaps": self._chart_top_overlaps(),
            },
        }

    def _chart_top_cooccurrences(self) -> BarChart:
        top = self.stats["top_cooccurrences"]
        return BarChart(
            type=ChartType.BAR.value,
            title="Top Class Co-occurrences",
            x=[f"{a} + {b}" for a, b, _ in top],
            y=[count for _, _, count in top],
            xlabel="Class Pair",
            ylabel="Co-occurrence Count",
        )

    def _chart_top_overlaps(self) -> BarChart:
        top = self.stats["top_overlaps"]
        return BarChart(
            type=ChartType.BAR.value,
            title="Top Class Overlaps (IoU > 0.2)",
            x=[f"{a} + {b}" for a, b, _ in top],
            y=[count for _, _, count in top],
            xlabel="Class Pair",
            ylabel="Overlap Count",
        )

    def _chart_no_cooccurrence_classes(self) -> BarChart:
        labels = self.stats["classes_with_no_cooccurrence"]
        return BarChart(
            type=ChartType.BAR.value,
            title="Classes with No Co-occurrence",
            x=labels,
            y=[0 for _ in labels],
            xlabel="Class",
            ylabel="Co-occurrence Count",
        )
