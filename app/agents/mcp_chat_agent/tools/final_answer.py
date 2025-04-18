from typing import Any

from pydantic_ai import Agent
from smolagents import Tool

from agents.ai_models import formatter_model
from agents.prompts.generator import PromptGenerator


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {
        "answer": {"type": "any", "description": "The final answer to the problem"}
    }
    output_type = "string"

    def __init__(
        self,
    ):
        super().__init__()
        analysis_goal = """
            Use `formatting` to ensure the final answer is:
            - Clear
            - Concise
            - Actionable
            - Well Markdown formatted
            - do not forget to add breaklines.
            - You can use emojis if it's relevant
            - Provide all the fix you fell needed, without changing the essence of the response.
            - Make it as short as possible, with a playful tone.
            - Make sure to make the relevant information in **bold** and suggest next steps.
            - Next step could be to Tag images, of Delete some shapes, or suggest to create a new dataset version.
            - If an action has been done, state it clearly.
            - If you want to display an URL, please show it as a mark down link, it's more beautiful.
            - THE LINK SHOULD OPEN A NEW TAB
        """
        prompt = PromptGenerator(context=True, analysis=True, format=True).generate(
            analysis_goal=analysis_goal, analysis_type=""
        )

        self.response_formatter = Agent(
            model=formatter_model, system_prompt=prompt, result_type=str
        )

    def forward(self, answer: Any) -> str:
        result = self.response_formatter.run(answer)
        return result.data
