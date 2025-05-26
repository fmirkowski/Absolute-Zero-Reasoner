from py_execution.exec import PythonExecutor
def validate_answer(answer, snippet, input_args, problem_type='code_o'):
    if problem_type.endswith('code_o'):
        exec = PythonExecutor(timeout=300)
        program = snippet + '\n' + f'f({input_args})'
        print(program)
        #determinism check
        code_answer1 = exec.execute(program=program)
        code_answer2 = exec.execute(program=program)
        non_det_penalty = -1 # ?
        if code_answer1 != code_answer2:
            return non_det_penalty
        print(code_answer1)
        if answer == code_answer1:
            return 1
        else: return 0


