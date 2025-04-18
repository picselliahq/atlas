from agents.annotation_agent.analysis.tightness.stats import TightnessStats
from agents.annotation_agent.models.actions import PossibleActions
from agents.annotation_agent.models.annotations import (
    TightnessShape,
)
from agents.common.interpreter.base_interpreter import BaseInterpreter
from agents.common.models.contents import ReportContent
from agents.common.models.shapes import (
    Label,
    LabeledBox,
    ShapeComparison,
    ShapeLabelComparisonElement,
)


class TightnessInterpreter(BaseInterpreter[TightnessStats, list[ReportContent]]):
    def __init__(
        self,
        stats: TightnessStats,
        min_treshold: float = 0.5,
        max_treshold: float = 0.5,
    ):
        super().__init__(stats)
        self.min_treshold = min_treshold
        self.max_treshold = max_treshold

    def interpret(self):
        pass

    def format(
        self,
        section: str,
        sub_section: str,
        agent_output=None,
    ) -> list[ReportContent]:
        contents: list[ReportContent] = []

        filtered_shapes: list[TightnessShape] = []
        for image in self.stats.shapes_tightness_issues:
            annotation = image.annotation
            for shape in annotation.shapes:
                if (
                    self.min_treshold < float(shape.iou)
                    and float(shape.iou) < self.max_treshold
                ):
                    filtered_shapes.append(shape)
        if filtered_shapes:
            chart_id = "chart-1"
            data = {
                chart_id: ShapeComparison(
                    type="shape-comparisons",
                    mode="shape-label",
                    data=[
                        ShapeLabelComparisonElement(
                            id=str(shape.id),
                            actual=LabeledBox(
                                x=shape.box[0],
                                y=shape.box[1],
                                w=shape.box[2],
                                h=shape.box[3],
                                label=Label(name=shape.label),
                            ),
                            expected=LabeledBox(
                                x=shape.suggestion[0],
                                y=shape.suggestion[1],
                                w=shape.suggestion[2],
                                h=shape.suggestion[3],
                                label=Label(name=shape.label),
                            ),
                        )
                        for shape in filtered_shapes
                    ],
                )
            }
            contents.append(
                ReportContent(
                    text="Identified tightness problems - This is Based on SAM Interactive segmentation.",
                    data=data,
                    section=section,
                    sub_section=sub_section,
                    name="Box Tightness",
                    potential_actions=[PossibleActions.TAG, PossibleActions.DELETE],
                )
            )
        return contents
