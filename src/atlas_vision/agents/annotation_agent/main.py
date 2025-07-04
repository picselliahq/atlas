import traceback
from typing import Any

import pandas as pd

from atlas_vision.agents.annotation_agent.common.get_context import PContext
from atlas_vision.agents.annotation_agent.tools import ALL_TOOLS
from atlas_vision.services.context import ContextService


def run_analysis(pctx: PContext) -> PContext:
    for tool in ALL_TOOLS:
        try:
            tool(pctx)
        except Exception:
            print(f"❌ Tool `{tool.__name__}` failed:")
            print(traceback.format_exc())
    pctx.sync()
    return pctx


class AnnotationAnalysisTool:
    def __init__(self, context_service: ContextService, **kwargs) -> None:
        super().__init__(**kwargs)
        self.context_service = context_service

    def forward(
        self, image_metadata: pd.DataFrame, annotation_metadata: pd.DataFrame
    ) -> dict[str, Any] | None:
        pctx = PContext(
            context_service=self.context_service,
            image_metadata=image_metadata,
            annotation_metadata=annotation_metadata,
        )
        pctx = run_analysis(pctx)
        return pctx.final_analysis
