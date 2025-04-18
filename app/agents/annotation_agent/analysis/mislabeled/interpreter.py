from agents.annotation_agent.analysis.mislabeled.stats import ShapeMislabeledStats
from agents.annotation_agent.models.annotations import (
    TightnessImage,
)
from agents.common.interpreter.analysis.base_interpreter import BaseInterpreter
from agents.common.models.actions import PossibleActions
from agents.common.models.contents import ReportContent
from agents.common.models.shapes import (
    Label,
    LabeledBox,
    ShapeComparison,
    ShapeLabelComparisonElement,
)


class WrongClassInterpreter(BaseInterpreter[ShapeMislabeledStats, ReportContent]):
    def interpret(self):
        pass

    def format(
        self, section: str, sub_section: str, content_name: str, agent_output=None
    ) -> ReportContent:
        wrongly_labeled_shape: list[TightnessImage] = self.stats.wrongly_labeled_shape
        chart_id = "chart-1"
        shape_data = []

        for image in wrongly_labeled_shape:
            if image.annotation:
                for shape in image.annotation.shapes:
                    shape_data.append(
                        ShapeLabelComparisonElement(
                            id=str(shape.id),
                            actual=LabeledBox(
                                label=Label(name=shape.label_annotated),
                                x=shape.x,
                                y=shape.y,
                                w=shape.w,
                                h=shape.h,
                            ),
                            expected=LabeledBox(
                                label=Label(name=shape.label_suggested),
                                x=shape.x,
                                y=shape.y,
                                w=shape.w,
                                h=shape.h,
                            ),
                        )
                    )

        if shape_data:
            data = {
                chart_id: ShapeComparison(
                    type="shape-comparisons", mode="label", data=shape_data
                )
            }
            text = (
                "Some objects appear to be annotated with the wrong class label. "
                "This is based on CLIP similarity comparisons."
            )
        else:
            data = {}
            text = "No mislabeled annotations were detected based on CLIP analysis."

        return ReportContent(
            text=text,
            data=data,
            section=section,
            sub_section=sub_section,
            name=content_name,
            potential_actions=[PossibleActions.UPDATE] if shape_data else [],
        )


class AmbiguousLabelsInterpreter(BaseInterpreter[ShapeMislabeledStats, ReportContent]):
    def interpret(self):
        pass

    def format(
        self, section: str, sub_section: str, content_name: str, agent_output=None
    ) -> ReportContent:
        ambiguous_annotations: list[TightnessImage] = self.stats.ambiguous_labeled_shape
        chart_id = "chart-1"
        shape_data = []

        for image in ambiguous_annotations:
            if image.annotation:
                for shape in image.annotation.shapes:
                    shape_data.append(
                        ShapeLabelComparisonElement(
                            id=str(shape.id),
                            actual=LabeledBox(
                                label=Label(name=shape.label_annotated),
                                x=shape.x,
                                y=shape.y,
                                w=shape.w,
                                h=shape.h,
                            ),
                            expected=LabeledBox(
                                label=Label(name=shape.label_suggested),
                                x=shape.x,
                                y=shape.y,
                                w=shape.w,
                                h=shape.h,
                            ),
                        )
                    )

        if shape_data:
            data = {
                chart_id: ShapeComparison(
                    type="shape-comparisons", mode="label", data=shape_data
                )
            }
            text = (
                "Some annotations show ambiguous class labels. "
                "CLIP embeddings suggest multiple plausible interpretations for these objects."
            )
        else:
            data = {}
            text = "No ambiguous annotations were detected based on CLIP analysis."

        return ReportContent(
            text=text,
            data=data,
            section=section,
            sub_section=sub_section,
            name=content_name,
            potential_actions=[PossibleActions.UPDATE] if shape_data else [],
        )
