from pathlib import Path


class PromptGenerator:
    def __init__(
        self,
        context: bool = True,
        analysis: bool = True,
        format: bool = True,
        lang: str = "ENGLISH",
    ):
        self.context = context
        self.analysis = analysis
        self.format = format
        self.lang = lang

    def generate(self, analysis_type: str, analysis_goal: str) -> str:
        prompt = """"""

        if self.context:
            md_path = Path(__file__).parent / "templates" / "analysis_context.md"
            with open(md_path) as file:
                prompt += file.read()

        if self.analysis:
            prompt += """
Here's what I'm about to do
            """
            prompt += analysis_type
            prompt += """
In order to identify:
            """
            prompt += analysis_goal

        if self.format:
            md_path = Path(__file__).parent / "templates" / "formatting_guidelines.md"
            with open(md_path) as file:
                prompt += file.read()

        prompt += f"""
I SPEAK IN {self.lang}, ALWAYS.
"""
        return prompt
