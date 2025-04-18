import pandas as pd

from agents.common.charts.types import ChartType, HistogramChart
from agents.common.stats.analysis.base_chart_stats import BaseChartStats


class ImageQualityStats(BaseChartStats["ImageQualityStats"]):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def _prepare_data(self) -> None:
        pass

    def _compute_stats(self) -> None:
        pass

    def _generate_all_chart_groups(self):
        self.chart_groups = {
            "image_statistics": {
                "blur_distribution": self._generate_blur_graph(),
                "contrast_distribution": self._generate_contrast_graph(),
                "luminance_distribution": self._generate_luminance_graph(),
            },
        }

    def _generate_luminance_graph(self) -> HistogramChart:
        return HistogramChart(
            type=ChartType.HIST,
            title="Luminance Distribution",
            bins=50,
            values=self.df["luminance"].tolist() if "luminance" in self.df else [],
            xlabel="Luminance",
            ylabel="Image Count",
        )

    def _generate_contrast_graph(self) -> HistogramChart:
        return HistogramChart(
            type=ChartType.HIST,
            title="Contrast Distribution",
            bins=50,
            values=self.df["contrast"].tolist() if "contrast" in self.df else [],
            xlabel="Contrast",
            ylabel="Image Count",
        )

    def _generate_blur_graph(self) -> HistogramChart:
        return HistogramChart(
            type=ChartType.HIST,
            title="Blur Score Distribution",
            bins=50,
            values=self.df["blur_score"].tolist() if "blur_score" in self.df else [],
            xlabel="Blur Score",
            ylabel="Image Count",
        )
