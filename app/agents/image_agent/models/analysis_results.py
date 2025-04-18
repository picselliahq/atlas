from pydantic import BaseModel, Field

from agents.image_agent.models.images import ImageGroup


class AnalysisResult(BaseModel):
    executive_summary: str = Field(
        description="A high level explanation on what type of information we get"
    )
    insights: str = Field(description="data-centric analysis of the luminance values")


class ExtendedAnalysisResult(AnalysisResult):
    value_analyzed: str = Field(description="")
    image_group: list[ImageGroup]
