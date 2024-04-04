import os

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from waterfall.eval import WinograndeEval
from waterfall.prompt import WinograndeBaselinePromptGenerator

load_dotenv()

hf_token = os.environ.get("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", load_in_8bit=True, token=hf_token
)
data = load_dataset("winogrande", "winogrande_xs", split="validation[:5]")

evaluator = WinograndeEval(
    model=model,
    tokenizer=tokenizer,
    dataset=data,
    prompt_generator=WinograndeBaselinePromptGenerator(),
)

print(evaluator.run())
