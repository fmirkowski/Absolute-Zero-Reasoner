import torch
from data.prompts import code_o_solver_prompt, instruction_following


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
        



def solver_pipeline(prompt: str, model, tokenizer, problem_type: str = "code_o") -> int:
    # 1. Generate response from LLM
    input_ids = tokenizer(prompt, return_tensors="pt")
    print('starting generation')
    with torch.no_grad():
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
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
    # return validate_answer(answer, ground_truth, problem_type)
    return answer




task_prompt = code_o_solver_prompt.format(snippet="""def f(x: int):
                                          return x**2""", input_args='3')
prompt = instruction_following.format(task_prompt)

