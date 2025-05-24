from dotenv import load_dotenv
load_dotenv()
from e2b_code_interpreter import Sandbox
import logging
 # By default the sandbox is alive for 5 minutes
# execution = sbx.run_code("print('hello world')") # Execute Python inside the sandbox
# print(execution.logs)

# files = sbx.files.list("/")
# print(files)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

class CodeExecutionError(Exception):
    """Exception raised when code execution fails in the sandbox."""
    pass
class PythonExecutor:
        def __init__(self, timeout: int = 300):
            self.timeout = timeout

        def execute(self, program: str):
            with Sandbox() as sbx:
                execution = sbx.run_code(program, timeout=self.timeout) # Execute Python inside the sandbox
                # print(execution.text)
                if getattr(execution, "error", None):
                    logger.error("Execution error: %s", execution.error)
                    raise CodeExecutionError(execution.error)
            
                return execution.text
  
# example:

# exec = PythonExecutor(timeout=300)
# print(exec.execute("""
# def s(x):
#     return x
# s(2)"""))
  
