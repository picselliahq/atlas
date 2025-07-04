import traceback

import pandas as pd

from atlas_vision.agents.image_agent.common.get_context import PContext
from atlas_vision.agents.image_agent.tools import ALL_TOOLS
from atlas_vision.services.context import ContextService
from atlas_vision.services.local_context import LocalContextService


class ImageAnalyticsTool:
    def __init__(
        self, context_service: ContextService | LocalContextService, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.context_service = context_service

    def forward(self, image_metadata: pd.DataFrame):
        pctx = PContext(
            context_service=self.context_service, image_metadata=image_metadata
        )
        for tool in ALL_TOOLS:
            try:
                tool(pctx)
            except Exception:
                print(f"❌ Tool `{tool.__name__}` failed:")
                print(traceback.format_exc())
        pctx.sync()
        return pctx.df
