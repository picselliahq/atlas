from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic_ai import Agent

StatsType = TypeVar("StatsType")
OutputType = TypeVar("OutputType")


class BaseGroupInterpreter(ABC, Generic[StatsType, OutputType]):
    def __init__(self, stats: StatsType, agent: Agent | None = None):
        self.stats = stats
        self.agent = agent

    @abstractmethod
    def interpret_group(self, group_name: str, prompt: str | None = None) -> Any: ...

    @abstractmethod
    def format_group_output(
        self,
        agent_output: Any,
        section: str,
        sub_section: str,
        name: str,
    ) -> OutputType: ...

    def interpret_group_to_content(
        self,
        group_name: str,
        section: str,
        sub_section: str,
        name: str,
        prompt: str | None = None,
    ) -> OutputType:
        agent_output = self.interpret_group(group_name=group_name, prompt=prompt)
        return self.format_group_output(
            agent_output=agent_output,
            section=section,
            sub_section=sub_section,
            name=name,
        )
