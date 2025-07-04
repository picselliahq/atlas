from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic_ai import Agent

StatsType = TypeVar("StatsType")
OutputType = TypeVar("OutputType")


class BaseInterpreter(ABC, Generic[StatsType, OutputType]):
    def __init__(self, stats: StatsType, agent: Agent | None = None):
        self.stats = stats
        self.agent = agent

    def run(self, section: str, sub_section: str, content_name: str) -> OutputType:
        agent_output = self.interpret()
        return self.format(
            section=section,
            sub_section=sub_section,
            content_name=content_name,
            agent_output=agent_output,
        )

    @abstractmethod
    def interpret(self) -> Any | None: ...

    @abstractmethod
    def format(
        self,
        section: str,
        sub_section: str,
        content_name: str,
        agent_output: Any | None,
    ) -> OutputType: ...
