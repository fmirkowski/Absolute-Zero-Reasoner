from py_execution.exec import PythonExecutor
def validate_answer(answer, ground_truth, snippet, input_args, problem_type='code_o'):
    if problem_type.endswith('code_o'):
        exec = PythonExecutor(timeout=300)
        program = snippet + '\n' + f'f({input_args})'
        print(program)
        code_answer = exec.execute(program=program)
        print(code_answer)
        if answer == code_answer:
            return 1


