from typing import Any

import pandas as pd
import picsellia

from agents.annotation_agent.analysis.tightness.sam_processor import SAMProcessor
from agents.annotation_agent.models.annotations import TightnessImage
from agents.common.charts.types import (
    BoxPlotChart,
    ChartType,
)
from agents.common.stats.analysis.base_chart_stats import BaseChartStats


class TightnessStats(BaseChartStats["TightnessStats"]):
    def __init__(
        self,
        df: pd.DataFrame,
        dataset: picsellia.DatasetVersion,
        asset_ids: list[str] | None = None,
    ):
        super().__init__(df=df)
        self.asset_ids = asset_ids

        self.dataset = dataset

        self.SAMProcessor = SAMProcessor()

        self.temp_df = df.copy(deep=True)
        self.temp_df["suggested_nx"] = pd.NA
        self.temp_df["suggested_ny"] = pd.NA
        self.temp_df["suggested_nw"] = pd.NA
        self.temp_df["suggested_nh"] = pd.NA
        self.temp_df["iou"] = pd.NA
        self.temp_df["image_width"] = pd.NA
        self.temp_df["image_height"] = pd.NA

        self.shapes_tightness_issues: list[TightnessImage] | None = None
        self.global_stats: dict[str, object] = {}
        self.per_class_stats: dict[str, dict[str, float]] = {}
        self.stats: dict[str, Any] = {}

    def _compute_stats(self) -> None:
        self._compute_global_stats()
        self._compute_per_class_tightness_stats()

    def _prepare_data(self):
        self.shapes_tightness_issues = self._run_on_dataset()
        median_iou = self.temp_df["iou"].median()
        self.temp_df = self.temp_df[self.temp_df["iou"] >= median_iou]
        self.temp_df["nx"] = self.temp_df["x"] / self.temp_df["image_width"]
        self.temp_df["ny"] = self.temp_df["y"] / self.temp_df["image_height"]
        self.temp_df["nw"] = self.temp_df["w"] / self.temp_df["image_width"]
        self.temp_df["nh"] = self.temp_df["h"] / self.temp_df["image_height"]
        self.temp_df["nx1"] = self.temp_df["nx"]
        self.temp_df["ny1"] = self.temp_df["ny"]
        self.temp_df["nx2"] = self.temp_df["nx"] + self.temp_df["nw"]
        self.temp_df["ny2"] = self.temp_df["ny"] + self.temp_df["nh"]
        self.temp_df["suggested_nx1"] = self.temp_df["suggested_nx"]
        self.temp_df["suggested_ny1"] = self.temp_df["suggested_ny"]
        self.temp_df["suggested_nx2"] = (
            self.temp_df["suggested_nx"] + self.temp_df["suggested_nw"]
        )
        self.temp_df["suggested_ny2"] = (
            self.temp_df["suggested_ny"] + self.temp_df["suggested_nh"]
        )
        self.temp_df["delta_nx1"] = self.temp_df["nx1"] - self.temp_df["suggested_nx1"]
        self.temp_df["delta_ny1"] = self.temp_df["ny1"] - self.temp_df["suggested_ny1"]
        self.temp_df["delta_nx2"] = self.temp_df["nx2"] - self.temp_df["suggested_nx2"]
        self.temp_df["delta_ny2"] = self.temp_df["ny2"] - self.temp_df["suggested_ny2"]
        self.temp_df["delta_area"] = (self.temp_df["nw"] * self.temp_df["nh"]) - (
            self.temp_df["suggested_nw"] * self.temp_df["suggested_nh"]
        )

    def _compute_global_stats(
        self,
    ):
        global_stats = (
            self.temp_df[
                ["delta_area", "delta_nx1", "delta_ny1", "delta_nx2", "delta_ny2"]
            ]
            .mean()
            .to_dict()
        )
        self.global_stats = {
            "tightness_score": 1 - global_stats["delta_area"],
            "delta_x1": global_stats["delta_nx1"],
            "delta_y1": global_stats["delta_ny1"],
            "delta_x2": global_stats["delta_nx2"],
            "delta_y2": global_stats["delta_ny2"],
        }

    def _compute_per_class_tightness_stats(self):
        grouped = self.temp_df.groupby("label")
        agg_means = grouped.agg(
            {
                "delta_area": "mean",
                "delta_nx1": "mean",
                "delta_ny1": "mean",
                "delta_nx2": "mean",
                "delta_ny2": "mean",
            }
        )
        for label, _ in grouped:
            self.per_class_stats[label] = {
                "tightness_score": 1 - agg_means.loc[label, "delta_area"],
                "delta_x1": agg_means.loc[label, "delta_nx1"],
                "delta_y1": agg_means.loc[label, "delta_ny1"],
                "delta_x2": agg_means.loc[label, "delta_nx2"],
                "delta_y2": agg_means.loc[label, "delta_ny2"],
            }

    def _generate_all_chart_groups(self):
        charts = {
            "delta_x1": self._charts_delta_x1(),
            "delta_y1": self._charts_delta_y1(),
            "delta_x2": self._charts_delta_x2(),
            "delta_y2": self._charts_delta_y2(),
        }

        self.chart_groups = {
            "tightness": {k: v for k, v in charts.items() if v is not None}
        }

    def _charts_delta_x1(self) -> BoxPlotChart | None:
        grouped = self.temp_df.groupby("label")["delta_nx1"].apply(list)
        if grouped.empty or all(len(v) == 0 for v in grouped.values):
            return None

        return BoxPlotChart(
            type=ChartType.BOX.value,
            title="Δx1 box annotation coordinates by Label",
            categories=grouped.index.tolist(),
            values=grouped.values.tolist(),
            xlabel="Class",
            ylabel="Δx1 (annotated x1 - suggested x1)",
        )

    def _charts_delta_y1(self) -> BoxPlotChart | None:
        grouped = self.temp_df.groupby("label")["delta_ny1"].apply(list)
        if grouped.empty or all(len(v) == 0 for v in grouped.values):
            return None

        return BoxPlotChart(
            type=ChartType.BOX.value,
            title="Δy1 box annotation coordinates by Label ",
            categories=grouped.index.tolist(),
            values=grouped.values.tolist(),
            xlabel="Class",
            ylabel="Δy1 (annotated y1 - suggested y1)",
        )

    def _charts_delta_x2(self) -> BoxPlotChart | None:
        grouped = self.temp_df.groupby("label")["delta_nx2"].apply(list)
        if grouped.empty or all(len(v) == 0 for v in grouped.values):
            return None

        return BoxPlotChart(
            type=ChartType.BOX.value,
            title="Δx2 box annotation coordinates by Label ",
            categories=grouped.index.tolist(),
            values=grouped.values.tolist(),
            xlabel="Class",
            ylabel="Δx2 (annotated x2 - suggested x2)",
        )

    def _charts_delta_y2(self) -> BoxPlotChart | None:
        grouped = self.temp_df.groupby("label")["delta_ny2"].apply(list)
        if grouped.empty or all(len(v) == 0 for v in grouped.values):
            return None

        return BoxPlotChart(
            type=ChartType.BOX.value,
            title="Δy2 box annotation coordinates by Label ",
            categories=grouped.index.tolist(),
            values=grouped.values.tolist(),
            xlabel="Class",
            ylabel="Δy2 (annotated y2 - suggested y2)",
        )

    def _run_on_dataset(
        self,
    ) -> list[TightnessImage]:
        asset_ids = self.asset_ids or self.df["asset_id"].unique().tolist()
        print(f"🔍 Running SAM on {len(asset_ids)} images...")
        results = []
        for asset_id in asset_ids:
            asset = self.dataset.find_asset(id=asset_id)
            result: TightnessImage = self.SAMProcessor.analyse_image(
                asset=asset,
            )
            if result:
                results.append(result)
                for shape in result.annotation.shapes:
                    mask = self.temp_df["shape_id"].eq(str(shape.id))
                    if mask.any():
                        self.temp_df.loc[mask, "suggested_nx"] = (
                            shape.suggestion[0] / asset.width
                        )
                        self.temp_df.loc[mask, "suggested_ny"] = (
                            shape.suggestion[1] / asset.height
                        )
                        self.temp_df.loc[mask, "suggested_nw"] = (
                            shape.suggestion[2] / asset.width
                        )
                        self.temp_df.loc[mask, "suggested_nh"] = (
                            shape.suggestion[3] / asset.height
                        )
                        self.temp_df.loc[mask, "iou"] = shape.iou
                        self.temp_df.loc[mask, "image_width"] = asset.width
                        self.temp_df.loc[mask, "image_height"] = asset.height
        # self.temp_df.to_csv("temp.csv")
        return results
