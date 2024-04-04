from dataclasses import dataclass

from transformers import TextGenerationPipeline

from .base import EvalPipeline


@dataclass(kw_only=True)
class WinograndeEval(EvalPipeline):
    def __post_init__(self):
        assert "sentence" in self.dataset.features, "Dataset must contain sentence"
        assert "option1" in self.dataset.features, "Dataset must contain option1"
        assert "option2" in self.dataset.features, "Dataset must contain option2"
        assert "answer" in self.dataset.features, "Dataset must contain answer"

        self.generator = TextGenerationPipeline(
            self.model, self.tokenizer, max_new_tokens=1, return_full_text=False
        )

    def run(self):
        preds = []
        for row in self.dataset:
            preds.append(self.evaluate_row(row))

        return self.calculate_metrics(preds, self.dataset["answer"])

    def evaluate_row(self, row: dict[str, str]) -> str:
        assert "sentence" in row, "Row must contain a sentence"
        assert "option1" in row, "Row must contain an option1"
        assert "option2" in row, "Row must contain a option2"
        # assert "answer" in row, "Row must contain a answer"

        prompt = self.prompt_generator(row)
        output = self.generator(prompt)[0]["generated_text"].strip()
        return output

    def calculate_metrics(self, preds: list[str], labels: list[str]):
        return "Not implemented yet"
