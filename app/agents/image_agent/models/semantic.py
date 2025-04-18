from pydantic import BaseModel, ConfigDict, Field

from agents.common.models.actions import PossibleActions


class SemanticImage(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    asset_id: str
    potential_actions: list[PossibleActions]


class SemanticGroup(BaseModel):
    group_name: str = Field(description="a 3 words semantic group name.")
    insights: str = Field(
        description="This is where you should say if the group is relevant for the dataset"
    )
    named_entities: list[str] = Field(
        description="Potential labels associated with this semantic group, suggestion for pre-labeling"
    )
    group_id: str = Field(description="The json key sent")


class ExtendedSemanticGroup(SemanticGroup):
    images: list[SemanticImage]
