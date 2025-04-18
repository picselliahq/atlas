import traceback
from typing import Any

import pandas as pd
from picsellia import DatasetVersion

from agents.annotation_agent.common.get_context import PContext
from agents.annotation_agent.tools import ALL_TOOLS
from services.context import ContextService


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
        self.client = context_service.client
        self.dataset: DatasetVersion = self.client.get_dataset_version_by_id(
            context_service.dataset_id
        )

    def forward(self, image_metadata: pd.DataFrame) -> dict[str, Any] | None:
        pctx = PContext(
            context_service=self.context_service, image_metadata=image_metadata
        )
        pctx = run_analysis(pctx)
        return pctx.final_analysis
