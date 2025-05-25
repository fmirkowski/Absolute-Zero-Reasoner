from datasets import load_dataset

# For HumanEval
humaneval_ds = load_dataset("openai_humaneval")
print(humaneval_ds['test'][0]['test'])