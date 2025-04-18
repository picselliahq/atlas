from abc import ABC, abstractmethod
from typing import TypeVar

import pandas as pd

from agents.common.charts.types import ChartData, ChartRenderer
from agents.common.stats.analysis.base_stats import BaseStats

T = TypeVar("T", bound="BaseChartStats")


class BaseChartStats(BaseStats[T], ABC):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

        self.chart_groups: dict[str, dict[str, ChartData]] = {}
        self.plot_groups: dict[str, dict[str, bytes]] = {}

    @abstractmethod
    def _prepare_data(self) -> None:
        pass

    @abstractmethod
    def _compute_stats(self) -> None:
        pass

    @abstractmethod
    def _generate_all_chart_groups(self) -> None:
        """Fill self.chart_groups with chart groups and charts per group."""
        pass

    def compute(self) -> T:
        self._prepare_data()
        self._compute_stats()
        self._generate_all_chart_groups()
        self._generate_all_plot_groups()
        return self

    def _generate_all_plot_groups(self) -> None:
        """Convert each chart in each group to PNG."""
        renderer = ChartRenderer()
        self.plot_groups = {}

        for group_name, charts in self.chart_groups.items():
            self.plot_groups[group_name] = {
                f"{chart_name}.png": renderer.to_png(chart)
                for chart_name, chart in charts.items()
            }
