import pandas as pd

from agents.common.charts.types import (
    BarChart,
    BoxPlotChart,
    ChartType,
    HistogramChart,
    ScatterChart,
)
from agents.common.stats.analysis.base_chart_stats import BaseChartStats


class ShapeStats(BaseChartStats["ShapeStats"]):
    def __init__(
        self,
        df: pd.DataFrame,
        small_area_thresh: float = 20,
        large_rel_area_thresh: float = 0.5,
    ):
        super().__init__(df)

        self.small_area_thresh = small_area_thresh
        self.large_rel_area_thresh = large_rel_area_thresh

        self.per_class_stats: dict[str, dict[str, float]] = {}
        self.global_stats: dict[str, float] = {}
        self.aspect_ratio_stats: dict[str, float] = {}
        self.shape_density_stats: dict[str, float] = {}

    def _prepare_data(self):
        if "x" in self.df.columns and "y" in self.df.columns:
            self.df["area"] = self.df["w"] * self.df["h"]
            self.df["image_area"] = self.df["image_width"] * self.df["image_height"]
            self.df["relative_area"] = self.df["area"] / self.df["image_area"]
            self.df["aspect_ratio"] = self.df["w"] / self.df["h"]
            self.df["w_norm"] = self.df["w"] / self.df["image_width"]
            self.df["h_norm"] = self.df["h"] / self.df["image_height"]
            self.df["area_norm"] = self.df["area"] / self.df["image_area"]
            self.df["center_x"] = (self.df["x"] + self.df["w"] / 2) / self.df[
                "image_width"
            ]
            self.df["center_y"] = (self.df["y"] + self.df["h"] / 2) / self.df[
                "image_height"
            ]
        else:
            pass

    def _compute_stats(self):
        if "x" in self.df.columns and "y" in self.df.columns:
            self._compute_global_stats()
            self._compute_per_class_stats()
            self._compute_aspect_ratio_stats()
            self._compute_shape_density_stats()
        else:
            self._compute_class_distribution()

    def _compute_global_stats(self):
        small_objects = self.df[self.df["area"] < self.small_area_thresh]
        large_objects = self.df[self.df["relative_area"] > self.large_rel_area_thresh]

        self.global_stats = {
            "num_shapes": len(self.df),
            "num_classes": self.df["label"].nunique(),
            "percent_small_objects": 100 * len(small_objects) / len(self.df),
            "percent_large_objects": 100 * len(large_objects) / len(self.df),
            "mean_aspect_ratio": self.df["aspect_ratio"].mean(),
            "small_area_thresh": self.small_area_thresh,
            "large_rel_area_thresh": self.large_rel_area_thresh,
        }

    def _compute_per_class_stats(self):
        self.per_class_stats = (
            self.df.groupby("label")["area"]
            .agg(["count", "mean", "median", "std"])
            .rename(
                columns={
                    "count": "count",
                    "mean": "mean_area",
                    "median": "median_area",
                    "std": "std_area",
                }
            )
            .to_dict(orient="index")
        )

    def _compute_aspect_ratio_stats(self):
        self.aspect_ratio_stats = {
            "min": self.df["aspect_ratio"].min(),
            "max": self.df["aspect_ratio"].max(),
            "std": self.df["aspect_ratio"].std(),
        }

    def _compute_shape_density_stats(self):
        shape_counts = self.df.groupby("asset_id")["shape_id"].count()
        self.shape_density_stats = {
            "mean": shape_counts.mean(),
            "std": shape_counts.std(),
            "min": shape_counts.min(),
            "max": shape_counts.max(),
            "median": shape_counts.median(),
            "num_empty_images": int((shape_counts == 0).sum()),
        }

    def _compute_class_distribution(self):
        counts = self.df["label"].value_counts()
        self.global_stats = {
            "num_classes": self.df["label"].nunique(),
            "num_shapes": len(self.df),
            "class_distribution": counts.to_dict(),
        }

    def _generate_all_chart_groups(self):
        if "x" in self.df.columns and "y" in self.df.columns:
            self.chart_groups = {
                "class_frequency": {
                    "class_distribution": self._chart_class_distribution(),
                },
                "object_analysis": {
                    "shape_density_distribution": self._chart_shape_density_distribution(),
                    "area_distribution": self._chart_area_distribution(),
                    "area_by_class": self._chart_area_by_class_distribution(),
                    "aspect_ratio_by_class": self._chart_aspect_ratio_by_class(),
                },
            }
        else:
            self.chart_groups = {
                "class_frequency": {
                    "class_distribution": self._chart_class_distribution(),
                }
            }

    def _chart_area_distribution(self) -> HistogramChart:
        return HistogramChart(
            type=ChartType.HIST.value,
            title="Distribution of Object Areas",
            values=self.df["area"].tolist(),
            bins=50,
            xlabel="Area (pixels)",
            ylabel="Count",
        )

    def _chart_class_distribution(self) -> BarChart:
        counts = self.df["label"].value_counts()
        return BarChart(
            type=ChartType.BAR.value,
            title="Object Count per Class",
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            xlabel="Class",
            ylabel="Count",
        )

    def _chart_area_by_class_distribution(self) -> BoxPlotChart:
        grouped = self.df.groupby("label")["area_norm"].apply(list)
        return BoxPlotChart(
            type=ChartType.BOX.value,
            title="Normalized Object Area Distribution per Class",
            categories=grouped.index.tolist(),
            values=grouped.values.tolist(),
            xlabel="Class",
            ylabel="Relative Area",
        )

    def _chart_aspect_ratio_by_class(self) -> BoxPlotChart:
        grouped = self.df.groupby("label")["aspect_ratio"].apply(list)
        return BoxPlotChart(
            type=ChartType.BOX.value,
            title="Aspect Ratio Distribution per Class",
            categories=grouped.index.tolist(),
            values=grouped.values.tolist(),
            xlabel="Class",
            ylabel="Aspect Ratio (w/h)",
        )

    def _chart_shape_density_distribution(self) -> HistogramChart:
        shape_counts = self.df.groupby("asset_id")["shape_id"].count()
        return HistogramChart(
            type=ChartType.HIST.value,
            title="Shape Density Distribution (Shapes per Image)",
            values=shape_counts.tolist(),
            bins=30,
            xlabel="Number of Shapes per Image",
            ylabel="Image Count",
        )

    def _chart_object_center_scatter_by_class(self) -> ScatterChart:
        return ScatterChart(
            type=ChartType.SCATTER.value,
            title="Object Center Positions by Class",
            x=self.df["center_x"].tolist(),
            y=self.df["center_y"].tolist(),
            color=self.df["label"].tolist(),
            xlabel="X Center (normalized)",
            ylabel="Y Center (normalized)",
        )
