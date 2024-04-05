from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class PromptGeneratorBase(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self) -> str:
        pass


@dataclass(kw_only=True)
class Pipeline(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self):
        pass


@dataclass(kw_only=True)
class EvalPipeline(Pipeline):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    dataset: Dataset
    prompt_generator: Callable | PromptGeneratorBase  # or PromptGeneratorBase

    @abstractmethod
    def calculate_metrics(self):
        pass
