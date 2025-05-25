code_o_solver_prompt = """
# Task: Deduce the Output of a Python Code Snippet Given the Code and Input
Given the following Code Snippet and the Input, think step by step then deduce the output that will be produced from plugging the Input into the Code Snippet. Put your output in ```output``` tags. Remember if the output is a string, wrap it in quotes. If the function returns multiple values, remember to use a tuple to wrap them.

# Code Snippet:
```python
{snippet}
```

# Input:
```input
{input_args}
```

# Example Output:
```output
{{'age': 20, 'city': 'New York'}}
```
"""

instruction_following = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {}\nAssistant: <think>"

# boxed_instruction = "{}\nPlease reason step by step, and put your final answer within \\boxed{{}}." this is ofr math LaTeX