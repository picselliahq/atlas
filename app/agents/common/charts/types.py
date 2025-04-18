import io
from enum import Enum
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from pydantic import BaseModel


class ChartType(str, Enum):
    BAR = "bar-chart"
    LINE = "line-chart"
    BOX = "boxplot-chart"
    HIST = "histogram-chart"
    HEATMAP = "heatmap-chart"
    SCATTER = "scatter-chart"
    TABLE = "table-chart"


class BaseChart(BaseModel):
    type: ChartType
    title: str = ""

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_png(self) -> bytes:
        return ChartRenderer.to_png(self)


class BarChart(BaseChart):
    type: Literal[ChartType.BAR]
    x: list[str]
    y: list[float]
    xlabel: str = ""
    ylabel: str = ""


class LineChart(BaseChart):
    type: Literal[ChartType.LINE]
    x: list[float]
    y: list[float]
    xlabel: str = ""
    ylabel: str = ""


class BoxPlotChart(BaseChart):
    type: Literal[ChartType.BOX]
    categories: list[str]
    values: list[list[float]]
    xlabel: str = ""
    ylabel: str = ""


class HistogramChart(BaseChart):
    type: Literal[ChartType.HIST]
    values: list[float]
    bins: int = 50
    xlabel: str = ""
    ylabel: str = ""


class HeatmapChart(BaseChart):
    type: Literal[ChartType.HEATMAP]
    matrix: list[list[int | float]]
    xlabels: list[str]
    ylabels: list[str]


class ScatterChart(BaseChart):
    type: Literal[ChartType.SCATTER]
    x: list[float]
    y: list[float]
    color: list[str]
    xlabel: str = ""
    ylabel: str = ""


class TableChart(BaseModel):
    type: Literal[ChartType.TABLE]
    rows: dict[str, str | int | float]


ChartData = (
    BarChart
    | LineChart
    | BoxPlotChart
    | HistogramChart
    | HeatmapChart
    | ScatterChart
    | TableChart
)


class ChartRenderer:
    @staticmethod
    def to_png(chart: BaseChart) -> bytes:
        fig, ax = plt.subplots(figsize=(10, 6))

        if isinstance(chart, BarChart):
            ax.barh(chart.x, chart.y, color="skyblue")
            ax.set_xlabel(chart.xlabel)
            ax.set_ylabel(chart.ylabel)
            ax.invert_yaxis()

        elif isinstance(chart, LineChart):
            ax.plot(chart.x, chart.y, marker="o")
            ax.set_xlabel(chart.xlabel)
            ax.set_ylabel(chart.ylabel)

        elif isinstance(chart, BoxPlotChart):
            ax.boxplot(chart.values, labels=chart.categories, vert=False)
            ax.set_xlabel(chart.ylabel)
            ax.set_ylabel(chart.xlabel)

        elif isinstance(chart, HistogramChart):
            ax.hist(chart.values, bins=chart.bins, color="orange")
            ax.set_xlabel(chart.xlabel)
            ax.set_ylabel(chart.ylabel)

        elif isinstance(chart, HeatmapChart):
            matrix = np.array(chart.matrix)
            all_ints = np.all(np.equal(np.mod(matrix, 1), 0))
            sns.heatmap(
                chart.matrix,
                xticklabels=chart.xlabels,
                yticklabels=chart.ylabels,
                ax=ax,
                annot=True,
                fmt="d" if all_ints else ".1f",
                cmap="coolwarm",
            )

        elif isinstance(chart, ScatterChart):
            labels = list(set(chart.color))
            label_to_idx = {label: i for i, label in enumerate(labels)}
            colors = [label_to_idx[label] for label in chart.color]
            scatter = ax.scatter(
                chart.x,
                chart.y,
                c=colors,
                cmap=cm.get_cmap("tab10", len(labels)),
                alpha=0.7,
            )

            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=label,
                    markerfacecolor=scatter.cmap(i),
                    markersize=8,
                )
                for label, i in label_to_idx.items()
            ]
            ax.legend(
                handles=handles,
                title="Class",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )

            ax.set_xlabel(chart.xlabel)
            ax.set_ylabel(chart.ylabel)

        ax.set_title(chart.title)

        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
