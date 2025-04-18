from typing import Literal

from pydantic import BaseModel

from agents.common.charts.display import RenderableChartData
from agents.common.models.actions import PossibleActions
from agents.common.models.assets import AssetComparison, Assets
from agents.common.models.shapes import ShapeComparison, Shapes

# === DATA BLOCKS ===


type DataBlock = (
    Assets | Shapes | ShapeComparison | AssetComparison | RenderableChartData
)


# === BASE CONTENT ===


class Content(BaseModel):
    name: str | None = None
    text: str
    data: dict[str, DataBlock] | None

    def to_report(
        self,
        *,
        section: str,
        sub_section: str,
        potential_actions: list[PossibleActions] | None = None,
    ) -> "ReportContent":
        return ReportContent(
            text=self.text,
            data=self.data,
            section=section,
            sub_section=sub_section,
            potential_actions=potential_actions,
        )


# === SPECIALIZED CONTENTS ===


class ReportContent(Content):
    section: str
    sub_section: str
    potential_actions: list[PossibleActions] | None


class ReportError(BaseModel):
    section: str | None = None
    sub_section: str | None = None
    message: str
    detail: str | None = None


class ReportEvent(BaseModel):
    event: Literal["computation_done"]
    section: str | None = None
    sub_section: str | None = None
    message: str


class ChatMessage(BaseModel):
    sender: Literal["human", "machine"]
    text: str
    data: str | None
    created_at: str
