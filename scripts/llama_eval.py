import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import AutoModelForCausalLM, AutoTokenizer

from waterfall.eval import WinograndeEval
from waterfall.prompt import WinograndeBaselinePromptGenerator

load_dotenv()

hf_token = os.environ.get("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", load_in_4bit=True, token=hf_token, device_map="cuda"
)
data = load_dataset("winogrande", "winogrande_xs", split="validation[:5]")

evaluator = WinograndeEval(
    model=model,
    tokenizer=tokenizer,
    dataset=data,
    prompt_generator=WinograndeBaselinePromptGenerator(),
)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

report, acc, cm = evaluator.run(
    output_dir=f"./output/llama/preds-{timestamp}.lst", output_dict=True
)
with open(f"./output/llama/report-{timestamp}.json", "w") as f:
    json.dump(report, f)

print(report)

with open(f"./output/llama/accuracy-{timestamp}.txt", "w") as f:
    f.write(str(acc))

print(f"Accuracy: {acc}")

print("Confusion matrix:")
print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels=["1", "2"])
disp.plot()
plt.savefig(f"./output/llama/confusion_matrix-{timestamp}.png")

print(f"Confusion matrix saved to ./output/llama/confusion_matrix-{timestamp}.png")
