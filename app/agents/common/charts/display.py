import math
from typing import Literal

import numpy as np

from .types import (
    BarChart,
    BaseChart,
    BoxPlotChart,
    ChartType,
    HeatmapChart,
    HistogramChart,
    LineChart,
    ScatterChart,
    TableChart,
)


class BoxPlotDisplayChart(BaseChart):
    type: Literal[ChartType.BOX]
    categories: list[str]
    values: list[list[float]]  # [low_whisker, Q1, median, Q3, high_whisker]
    outliers: list[list[float]]  # [index, value]
    xlabel: str = ""
    ylabel: str = ""

    @classmethod
    def from_raw(cls, chart: BoxPlotChart) -> "BoxPlotDisplayChart":
        boxplot_data: list[list[float]] = []
        outliers: list[list[float]] = []

        for idx, values in enumerate(chart.values):
            if not values:
                boxplot_data.append([math.nan] * 5)
                continue

            sorted_vals = np.sort(values)
            q1 = np.percentile(sorted_vals, 25)
            q3 = np.percentile(sorted_vals, 75)
            median = np.percentile(sorted_vals, 50)
            iqr = q3 - q1
            lower_whisker = max(min(sorted_vals), q1 - 1.5 * iqr)
            upper_whisker = min(max(sorted_vals), q3 + 1.5 * iqr)

            boxplot_data.append([lower_whisker, q1, median, q3, upper_whisker])

            for val in values:
                if val < lower_whisker or val > upper_whisker:
                    outliers.append([idx, val])

        return cls(
            title=chart.title,
            type=ChartType.BOX,
            categories=chart.categories,
            values=boxplot_data,
            outliers=outliers,
            xlabel=chart.xlabel,
            ylabel=chart.ylabel,
        )


class HistogramDisplayChart(BaseChart):
    type: Literal[ChartType.HIST]
    bin_edges: list[float]  # x coordinates (edges of bins)
    counts: list[int]  # y coordinates (counts per bin)
    xlabel: str = ""
    ylabel: str = ""

    @classmethod
    def from_raw(cls, chart: HistogramChart) -> "HistogramDisplayChart":
        if not chart.values:
            return cls(
                title=chart.title,
                type=ChartType.HIST,
                bin_edges=[],
                counts=[],
                xlabel=chart.xlabel,
                ylabel=chart.ylabel,
            )

        counts, bin_edges = np.histogram(chart.values, bins=chart.bins)

        return cls(
            title=chart.title,
            type=ChartType.HIST,
            bin_edges=bin_edges.tolist(),
            counts=counts.tolist(),
            xlabel=chart.xlabel,
            ylabel=chart.ylabel,
        )


RenderableChartData = (
    BarChart
    | LineChart
    | BoxPlotDisplayChart
    | HistogramDisplayChart
    | HeatmapChart
    | ScatterChart
    | TableChart
)
