from waterfall.prompt import WinograndeBaselinePromptGenerator

prompt = WinograndeBaselinePromptGenerator()
print(
    prompt.generate(
        {"sentence": "The cat sat on the", "option1": "mat", "option2": "chair"}
    )
)
