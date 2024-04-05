import os
from dataclasses import dataclass

from numpy import ndarray
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

    def run(
        self, output_dir: str | None = None, output_dict: bool = False
    ) -> tuple[str | dict, float, ndarray]:
        preds = []
        for row in self.dataset:
            preds.append(self.evaluate_row(row))

        if output_dir:
            if not os.path.exists(os.path.dirname(output_dir)):
                os.makedirs(os.path.dirname(output_dir))
            with open(output_dir, "w") as f:
                for pred in preds:
                    f.write(pred + "\n")

        print("Finished evaluating")
        print("Calculating metrics")
        print(self.dataset["answer"])
        print(preds)
        return (
            self.calculate_metrics(
                preds, self.dataset["answer"], output_dict=output_dict
            ),
            accuracy_score(self.dataset["answer"], preds),
            confusion_matrix(self.dataset["answer"], preds, labels=["1", "2"]),
        )

    def evaluate_row(self, row: dict[str, str]) -> str:
        assert "sentence" in row, "Row must contain a sentence"
        assert "option1" in row, "Row must contain an option1"
        assert "option2" in row, "Row must contain a option2"
        # assert "answer" in row, "Row must contain a answer"

        prompt = self.prompt_generator(row)
        output = self.generator(prompt)[0]["generated_text"].strip()
        return output

    def calculate_metrics(
        self, preds: list[str], answers: list[str], output_dict: bool = False
    ) -> str | dict:
        return classification_report(
            answers, preds, labels=["1", "2"], output_dict=output_dict
        )
