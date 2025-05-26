import torch
from data.prompts import code_o_solver_prompt, instruction_following
from py_execution.validate_answer import validate_answer

"""
PROMPT/TASK (from proposer) -> LLM soluton -> Python filters, construct valid reasoning questions (??) -> binary accuracy reward 

Goal first implement Deduction - infering output o from p(i)
questions:
    - what and how shoudl rpompt look like? - we have example from codebase
    - how should we extract it - in the output tags sopecuficed in pormpt
    - what about the <think> tags - ans - prompt insturciton following and then penalise in reward

Advices: keep stuff modular, like those code_i, ...
"""

def extract_answer(content, problem_type):
    """Extract content between <answer></answer> tags from the LLM response"""
    try:
        start_idx = content.find("<answer>") + len("<answer>")
        end_idx = content.find("</answer>")
        if start_idx == -1 or end_idx == -1:
            return None
        
        start_idx_think = content.find("<think>") + len("<think>")
        end_idx_think = content.find("</think>")
        if start_idx_think == -1 or end_idx_think == -1:
            return None
        think_content = content[start_idx_think:end_idx_think].strip()
        return content[start_idx:end_idx].strip()    
    except:
        return None

def extract_input_output(extracted_content, problem_type):
    try:
        start_idx = extracted_content.find("'''output") + len("'''output")
        end_idx = extracted_content[start_idx:].find("'''") # : after to get the last ones
        if start_idx == -1 or end_idx == -1:
            return None
        return extracted_content[start_idx:end_idx].strip()
    except:
        return None
        


def solver_pipeline(prompt: str, model, tokenizer, snippet, input_arg, problem_type: str = "code_o") -> int:
    # Determine device
    # We tak snipet and input_arg, this is specific for a deduction task
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 1. Generate response from LLM
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    print('starting')
    with torch.no_grad():
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Move output back to CPU for decoding
    output_ids = output_ids.cpu()
    
    # 2. Decode the response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generation = response.split(prompt)[-1].strip()
    print(f'generated response: {generation}')
    # 3. Extract answer based on problem type
    extracted_content = extract_answer(generation, problem_type=problem_type)  # how should the answer look like tho?
    if not extracted_content:
        return -1
    # 4. Parse and validate based on problem type
    if problem_type.endswith('code_o'):
        answer = extract_input_output(extracted_content, problem_type=problem_type)
    # elif problem_type.endswith('code_o'):
    #     answer = extract_input_output(extracted_content)
    # elif problem_type.endswith('code_f'):
    #     success, answer = parse_code_function(extracted_content)
    #     if not success:
    #         return 0
    
    # 5. Execute validation (you'll need a PythonExecutor instance)
    # This is where the binary reward is determined
    return validate_answer(answer, ground_truth, snippet, input_arg, problem_type)
    # return answer

# Specific for deduction not really modular, we can jusyt do :None later on
snippet = """def f(x: int):
    return x**2"""
input_args = '3'
task_prompt = code_o_solver_prompt.format(snippet=snippet, input_args=input_args)
prompt = instruction_following.format(task_prompt)
validate_answer(9, 1, snippet, input_args)
from transformers import AutoModelForCausalLM, AutoTokenizer
# from solver_pipeline import prompt, solver_pipeline
# Load model and tokenizer
# model_name = "Qwen/Qwen3-4B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# print(f'answer: {solver_pipeline(prompt, model, tokenizer, snippet, input_args)}')