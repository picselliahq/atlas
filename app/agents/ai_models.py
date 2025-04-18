from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from smolagents import LiteLLMModel

from config.settings import settings

report_model = (
    AnthropicModel(
        model_name=settings.report_model.name,
        provider=AnthropicProvider(
            api_key=settings.report_model.api_key,
        ),
    )
    if settings.report_model.name.startswith("claude")
    else OpenAIModel(
        model_name=settings.report_model.name,
        provider=OpenAIProvider(
            base_url=settings.report_model.base_url,
            api_key=settings.report_model.api_key,
        ),
    )
)

chat_model = LiteLLMModel(
    model_id=settings.chat_model.name,
    api_base=settings.chat_model.base_url,
    api_key=settings.chat_model.api_key,
    temperature=1,
)

formatter_model = OpenAIModel(
    model_name=settings.formatter_model.name,
    provider=OpenAIProvider(
        base_url=settings.formatter_model.base_url,
        api_key=settings.formatter_model.api_key,
    ),
)
