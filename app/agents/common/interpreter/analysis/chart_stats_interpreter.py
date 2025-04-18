from typing import TypeVar

from pydantic_ai import Agent, BinaryContent

from agents.common.charts.display import (
    BoxPlotDisplayChart,
    HistogramDisplayChart,
)
from agents.common.charts.types import BoxPlotChart, ChartData, HistogramChart
from agents.common.interpreter.analysis.base_group_interpreter import (
    BaseGroupInterpreter,
)
from agents.common.models.contents import ReportContent
from agents.common.stats.analysis.base_chart_stats import BaseChartStats

T = TypeVar("T", bound=BaseChartStats)


class ChartStatsInterpreter(BaseGroupInterpreter[T, ReportContent]):
    def __init__(self, stats: T, agent: Agent, prompt: str | None = None):
        super().__init__(stats=stats, agent=agent)
        self.prompt = prompt

    def interpret_group(
        self,
        group_name: str,
        prompt: str | None = None,
    ) -> list[tuple[str, ChartData, str]]:
        results = []
        charts = self.stats.chart_groups.get(group_name, {})
        plots = self.stats.plot_groups.get(group_name, {})

        effective_prompt = prompt or self.prompt

        for chart_name, chart in charts.items():
            image = plots.get(f"{chart_name}.png")
            if not image:
                continue

            input_payload = (
                [effective_prompt, BinaryContent(image, media_type="image/png")]
                if effective_prompt
                else [BinaryContent(image, media_type="image/png")]
            )
            result = self.agent.run_sync(input_payload)
            insight = result.data
            results.append((chart_name, chart, insight))

        return results

    def format_group_output(
        self,
        agent_output: list[tuple[str, ChartData, str]],
        section: str,
        sub_section: str,
        content_name: str,
        *args,
        **kwargs,
    ) -> ReportContent:
        full_text = ""
        data = {}
        chart_index = 1

        for _chart_name, chart, insight_text in agent_output:
            if isinstance(chart, BoxPlotChart):
                renderable = BoxPlotDisplayChart.from_raw(chart)
            elif isinstance(chart, HistogramChart):
                renderable = HistogramDisplayChart.from_raw(chart)
            else:
                renderable = chart

            chart_id = f"chart-{chart_index}"
            chart_index += 1

            full_text += f'{insight_text}\n<Data id="{chart_id}"/>\n'
            data[chart_id] = renderable.to_dict()

        return ReportContent(
            text=full_text,
            data=data,
            section=section,
            sub_section=sub_section,
            name=content_name,
            potential_actions=None,
        )

    def interpret_group_to_content(
        self,
        group_name: str,
        section: str,
        sub_section: str,
        content_name: str,
        prompt: str | None = None,
        *args,
        **kwargs,
    ) -> ReportContent:
        agent_output = self.interpret_group(group_name=group_name, prompt=prompt)
        return self.format_group_output(
            agent_output,
            section=section,
            sub_section=sub_section,
            content_name=content_name,
        )
