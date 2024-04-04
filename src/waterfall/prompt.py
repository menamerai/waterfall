from .base import PromptGeneratorBase
from .constants import winogrande_examples


class WinograndeBaselinePromptGenerator(PromptGeneratorBase):
    def __init__(self, examples: list[dict[str, str]] | None):
        if examples is None:
            examples = winogrande_examples

        self.prompt = "The following sentence is missing a word. Please select the option that best fits the blank.\n\n"
        for example in examples:
            self.prompt += f"Sentence: {example['sentence']}\n"
            self.prompt += f"Option 1: {example['option1']}\n"
            self.prompt += f"Option 2: {example['option2']}\n"
            self.prompt += f"Answer: {example['answer']}\n\n"

    def __call__(self, doc: dict[str, str]):
        prompt = self.prompt
        prompt += f"Sentence: {doc['sentence']}\n"
        prompt += f"Option 1: {doc['option1']}\n"
        prompt += f"Option 2: {doc['option2']}\n"
        prompt += "Answer: "
