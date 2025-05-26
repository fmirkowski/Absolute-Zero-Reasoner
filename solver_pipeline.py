import torch
from data.prompts import code_o_solver_prompt, instruction_following
from py_execution.validate_answer import validate_answer
from tqdm import tqdm
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
        # print(f'[INFO] Content passed to extract answer and think tags: {content}')
        start_idx = content.find("<answer>") + len("<answer>")
        end_idx = content.find("</answer>")
        if start_idx == -1 or end_idx == -1:
            print("[WARNING] Could not find answer tags in content")
            return None
        
        start_idx_think = content.find("<think>") + len("<think>")
        end_idx_think = content.find("</think>")
        if start_idx_think == -1 or end_idx_think == -1:
            print("[WARNING] Could not find think tags in content")
            return None
        think_content = content[start_idx_think:end_idx_think].strip()
        return content[start_idx:end_idx].strip()    
    except Exception as e:
        print(f"[ERROR] Error extracting answer: {str(e)}")
        return None

def extract_input_output(extracted_content, problem_type):
    try:
        start_idx = extracted_content.find("```output") + len("```output")
        end_idx = extracted_content[start_idx:].find("```") # : after to get the last ones
        if start_idx == -1 or end_idx == -1:
            print("[WARNING] Could not find output tags in content")
            return None
        return extracted_content[start_idx:end_idx].strip()
    except Exception as e:
        print(f"[ERROR] Error extracting input/output: {str(e)}")
        return None
        
def solver_pipeline(prompt: str, model, tokenizer, snippet, input_arg, problem_type: str = "code_o") -> int:
    # Determine device
    # We tak snipet and input_arg, this is specific for a deduction task
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = model.to(device)
    
    # 1. Generate response from LLM
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    print('[INFO] Starting LLM generation')
    with torch.no_grad():
        MAX_TOKENS = 256
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Move output back to CPU for decoding
    output_ids = output_ids.cpu()
    
    # 2. Decode the response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f'[INFO] THE WHOLE Generated response: {response}\n\n')

    generation = response.split(prompt)[-1].strip()
    generation = '<think>' + generation
    print(f'[INFO] Generated response: {generation}')
    
    # 3. Extract answer based on problem type
    extracted_content = extract_answer(generation, problem_type=problem_type)  # how should the answer look like tho?
    if not extracted_content:
        print("[WARNING] Failed to extract answer content, no answer tags")
        return -1
        
    # 4. Parse and validate based on problem type
    if problem_type.endswith('code_o'):
        answer = extract_input_output(extracted_content, problem_type=problem_type)
        print(f"[INFO] Extracted answer: {answer}")
        if not answer:
            return -1 # no output tags
    # elif problem_type.endswith('code_o'):
    #     answer = extract_input_output(extracted_content)
    # elif problem_type.endswith('code_f'):
    #     success, answer = parse_code_function(extracted_content)
    #     if not success:
    #         return 0
    
    # 5. Execute validation (you'll need a PythonExecutor instance)
    # This is where the binary reward is determined
    result = validate_answer(answer, snippet, input_arg, problem_type)
    print(f"[INFO] Validation result: {result}")
    return result
    # return answer

# Specific for deduction not really modular, we can jusyt do :None later on
snippet = """def f(x: int):
    return x**2"""
input_args = '3'
task_prompt = code_o_solver_prompt.format(snippet=snippet, input_args=input_args)
prompt = instruction_following.format(task_prompt)
from transformers import AutoModelForCausalLM, AutoTokenizer
# Load model and tokenizer
model_name = "Qwen/Qwen3-4B"
print(f"[INFO] Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
result = solver_pipeline(prompt, model, tokenizer, snippet, input_args)
print(f'[INFO] Final answer: {result}')