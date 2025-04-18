import json

from pydantic_ai.agent import AgentRunResult

from agents.common.interpreter.analysis.base_interpreter import BaseInterpreter
from agents.common.models.contents import ReportContent
from agents.common.utils import (
    dict_to_markdown_table,
)
from agents.image_agent.analysis.data_card.stats import DataCardStats
from agents.image_agent.models.datacard import DataCard


class DataCardInterpreter(BaseInterpreter[DataCardStats, ReportContent]):
    def interpret(self) -> AgentRunResult[DataCard]:
        if not self.agent:
            raise ValueError("Agent is not set for DataCardInterpreter")
        return self.agent.run_sync(json.dumps(self.stats.context))

    def format(
        self,
        section: str,
        sub_section: str,
        content_name: str,
        agent_output: AgentRunResult[DataCard],
    ) -> ReportContent:
        card: DataCard = agent_output.data

        table_values = {
            "Creator": card.creator,
            "Dataset name": card.name,
            "Version": card.version,
            "Description": card.description or "—",
            "Dataset size": card.dataset_size,
            "Task": card.task,
        }
        table = dict_to_markdown_table(table_values)

        extra_description = (
            f"\n{card.verbose_description.strip()}" if card.verbose_description else ""
        )
        text = (
            f"This dataset card provides key metadata and contextual information "
            f"to help better understand the dataset's purpose and structure. "
            f"Below is a summary of the main attributes:\n\n"
            f"{table}\n\n"
            f"{extra_description}"
        )

        return ReportContent(
            text=text,
            data=None,
            section=section,
            sub_section=sub_section,
            name=content_name,
            potential_actions=None,
        )
