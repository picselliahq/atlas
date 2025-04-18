from litellm import completion
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from smolagents import LiteLLMModel


class CityLocation(BaseModel):
    city: str
    country: str


UK = [
    "England",
    "Great Britain",
    "United Kingdom",
    "The United Kingdom of Great Britain and Northern Ireland",
]


def test_local_report_model(report_model: OpenAIModel) -> None:
    agent = Agent(
        model=report_model,
        result_type=CityLocation,
        system_prompt=(
            "Extract me the city and country from the text. "
            "if you can't extract it, return the text as is."
        ),
    )
    result = agent.run_sync("Where were the olympics held in 2012?")
    assert isinstance(result.data, CityLocation)
    assert result.data.city == "London"
    assert result.data.country in UK


def test_local_chat_model(chat_model: LiteLLMModel) -> None:
    result = completion(
        model=chat_model.model_id,
        messages=[{"role": "user", "content": "Where were the olympics held in 2012?"}],
        temperature=1,
        response_format=CityLocation,
        base_url=chat_model.api_base,
        api_key=chat_model.api_key,
    )
    response = CityLocation.model_validate_json(result.choices[0].message.content)
    assert response.city == "London"
    assert response.country in UK
