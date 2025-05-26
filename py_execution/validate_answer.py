from exec import PythonExecutor
def validate_answer(answer, ground_truth, snippet, input_args, problem_type='code_o'):
    if problem_type.endswith('code_o'):
        exec = PythonExecutor(timeout=300):
        program = snippet + '\n' + f'f({input_args})'
        code_answer = exec.execute(program=program)
        if answer == code_answer:
            return 1
