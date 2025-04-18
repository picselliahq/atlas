import logging
from typing import TypeVar

from pydantic_ai import Agent, BinaryContent

from agents.common.charts.display import (
    BoxPlotDisplayChart,
    HistogramDisplayChart,
)
from agents.common.charts.types import BarChart, BoxPlotChart, ChartData, HistogramChart
from agents.common.interpreter.analysis.base_group_interpreter import (
    BaseGroupInterpreter,
)
from agents.common.models.contents import ReportContent, ReportError
from agents.common.stats.analysis.base_chart_stats import BaseChartStats

T = TypeVar("T", bound=BaseChartStats)

logger = logging.getLogger(__name__)


class MultiChartStatsInterpreter(BaseGroupInterpreter[T, ReportContent]):
    def __init__(self, stats: T, agent: Agent, prompt: str | None = None):
        super().__init__(stats=stats, agent=agent)
        self.prompt = prompt

    def interpret_group(
        self, group_name: str, prompt: str | None = None
    ) -> tuple[str, dict[str, ChartData]]:
        charts = self.stats.chart_groups.get(group_name, {})
        plots = self.stats.plot_groups.get(group_name, {})

        images = [
            BinaryContent(plots[f"{chart_name}.png"], media_type="image/png")
            for chart_name in charts
            if f"{chart_name}.png" in plots
        ]
        if not images:
            return "", {}

        effective_prompt = prompt or self.prompt
        input_payload = [effective_prompt] + images if effective_prompt else images

        result = self.agent.run_sync(input_payload)
        insight = result.data.strip()
        return insight, charts

    def format_group_output(
        self,
        agent_output: tuple[str, dict[str, ChartData]],
        section: str,
        sub_section: str,
        name: str = "Chart Group Analysis",
    ) -> ReportContent:
        insight_text, charts = agent_output
        full_text = f"{insight_text}\n"
        data = {}
        chart_index = 1

        for _chart_name, chart in charts.items():
            if self._is_chart_empty(chart):
                continue
            if isinstance(chart, BoxPlotChart):
                renderable = BoxPlotDisplayChart.from_raw(chart)
            elif isinstance(chart, HistogramChart):
                renderable = HistogramDisplayChart.from_raw(chart)
            else:
                renderable = chart

            chart_id = f"chart-{chart_index}"
            data[chart_id] = renderable.to_dict()
            full_text += f'\n<Data id="{chart_id}"/>'
            chart_index += 1

        return ReportContent(
            name=name,
            section=section,
            sub_section=sub_section,
            text=full_text,
            data=data,
            potential_actions=None,
        )

    def interpret_group_to_content(
        self,
        group_name: str,
        section: str,
        sub_section: str,
        name: str = "Chart Group Analysis",
        prompt: str | None = None,
    ) -> ReportContent | ReportError:
        try:
            agent_output = self.interpret_group(group_name, prompt=prompt)
            return self.format_group_output(
                agent_output=agent_output,
                section=section,
                sub_section=sub_section,
                name=name,
            )
        except Exception as e:
            logger.warning(f"Fail to interpret group: {e}")
            return ReportError(
                section=section,
                sub_section=sub_section,
                message=f"Failed to interprete in group {group_name}.",
            )

    def _is_chart_empty(self, chart: ChartData) -> bool:
        if isinstance(chart, HistogramChart):
            return sum(chart.values) == 0
        elif isinstance(chart, BoxPlotChart):
            return all(len(v) == 0 for v in chart.values)
        elif isinstance(chart, BarChart):
            return all(y == 0 for y in chart.y)
        return False
