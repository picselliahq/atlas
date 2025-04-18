import yaml
from smolagents import PromptTemplates, ToolCallingAgent

from agents.ai_models import chat_model
from agents.mcp_chat_agent.tools.final_answer import FinalAnswerTool
from agents.mcp_chat_agent.tools.get_dataset_informations import (
    GetMetadatasetInformation,
)
from services.context import ContextService


def create_mcp_agent(context_service: ContextService):
    with open("agents/mcp_chat_agent/system_prompt/tool_calling_prompt.yaml") as f:
        tool_calling_prompt = yaml.safe_load(f)
    tool_calling_prompt = PromptTemplates(**tool_calling_prompt)
    get_dataset_info_tool = GetMetadatasetInformation(context=context_service)
    final_answer = FinalAnswerTool()
    return ToolCallingAgent(
        tools=[get_dataset_info_tool, final_answer],
        model=chat_model,
        name="MCP",
        max_steps=1,
        prompt_templates=tool_calling_prompt,
    )
