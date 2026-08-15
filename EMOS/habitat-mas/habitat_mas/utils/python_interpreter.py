import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List

from colorama import Fore


@dataclass(frozen=True)
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


class SubprocessInterpreter:
    r"""SubprocessInterpreter is a class for executing code files or code
    strings in a subprocess.

    This class handles the execution of code in different scripting languages
    (currently Python and Bash) within a subprocess, capturing their
    stdout and stderr streams, and allowing user checking before executing code
    strings.

    Args:
        require_confirm (bool, optional): If True, prompt user before running
            code strings for security. (default: :obj:`True`)
        print_stdout (bool, optional): If True, print the standard output of
            the executed code. (default: :obj:`False`)
        print_stderr (bool, optional): If True, print the standard error of the
            executed code. (default: :obj:`True`)
    """

    _CODE_EXECUTE_CMD_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python {file_name}",
        "bash": "bash {file_name}",
    }

    _CODE_EXTENSION_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "py",
        "bash": "sh",
    }

    _CODE_TYPE_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python",
        "py3": "python",
        "python3": "python",
        "py": "python",
        "shell": "bash",
        "bash": "bash",
        "sh": "bash",
    }

    def __init__(
        self,
        require_confirm: bool = False,
        print_stdout: bool = False,
        print_stderr: bool = True,
    ) -> None:
        self.require_confirm = require_confirm
        self.print_stdout = print_stdout
        self.print_stderr = print_stderr

    def run_file(
        self,
        file: Path,
        code_type: str,
    ) -> str:
        r"""Executes a code file in a subprocess and captures its output.

        Args:
            file (Path): The path object of the file to run.
            code_type (str): The type of code to execute (e.g., 'python',
                'bash').

        Returns:
            str: A string containing the captured stdout and stderr of the
                executed code.

        Raises:
            RuntimeError: If the provided file path does not point to a file.
            InterpreterError: If the code type provided is not supported.
        """
        if not file.is_file():
            raise RuntimeError(f"{file} is not a file.")
        code_type = self._check_code_type(code_type)
        cmd = shlex.split(
            self._CODE_EXECUTE_CMD_MAPPING[code_type].format(file_name=str(file))
        )
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = proc.communicate()
        if self.print_stdout and stdout:
            print("======stdout======")
            print(Fore.GREEN + stdout + Fore.RESET)
            print("==================")
        if self.print_stderr and stderr:
            print("======stderr======")
            print(Fore.RED + stderr + Fore.RESET)
            print("==================")
        exec_result = f"{stdout}"
        exec_result += f"(stderr: {stderr})" if stderr else ""
        return exec_result

    def run(
        self,
        code: str,
        code_type: str,
    ) -> str:
        r"""Generates a temporary file with the given code, executes it, and
            deletes the file afterward.

        Args:
            code (str): The code string to execute.
            code_type (str): The type of code to execute (e.g., 'python',
                'bash').

        Returns:
            str: A string containing the captured stdout and stderr of the
                executed code.

        Raises:
            InterpreterError: If the user declines to run the code or if the
                code type is unsupported.
        """
        code_type = self._check_code_type(code_type)

        if code_type == "python":
            # Ignore warnings in Python code
            code = 'import warnings\nwarnings.filterwarnings("ignore")\n' + code

        # Print code for security checking
        if self.require_confirm:
            print(f"The following {code_type} code will run on your computer:")
            print(Fore.CYAN + code + Fore.RESET)
            while True:
                choice = input("Running code? [Y/n]:").lower()
                if choice in ["y", "yes", "ye", ""]:
                    break
                elif choice in ["no", "n"]:
                    raise ValueError(
                        "Execution halted: User opted not to run the code. "
                        "This choice stops the current operation and any "
                        "further code execution."
                    )
        temp_file_path = self._create_temp_file(
            code=code, extension=self._CODE_EXTENSION_MAPPING[code_type]
        )

        result = self.run_file(temp_file_path, code_type)

        temp_file_path.unlink()
        return result

    def run_python_detailed(
        self, code: str, timeout_seconds: float = 10
    ) -> ExecutionResult:
        """Execute generated Python with validation, timeout, and structured output."""
        try:
            compile(code, "<qwen-generated>", "exec")
        except SyntaxError as error:
            return ExecutionResult(
                returncode=1,
                stdout="",
                stderr=f"SyntaxError: line {error.lineno}: {error.msg}",
                timed_out=False,
            )

        executable_code = (
            'import warnings\nwarnings.filterwarnings("ignore")\n' + code
        )
        temp_file_path = self._create_temp_file(
            code=executable_code,
            extension=self._CODE_EXTENSION_MAPPING["python"],
        )
        try:
            cmd = shlex.split(
                self._CODE_EXECUTE_CMD_MAPPING["python"].format(
                    file_name=str(temp_file_path)
                )
            )
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                return ExecutionResult(
                    returncode=proc.returncode if proc.returncode is not None else -1,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=True,
                )
            return ExecutionResult(
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
            )
        finally:
            temp_file_path.unlink(missing_ok=True)


    def _create_temp_file(self, code: str, extension: str) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=f".{extension}"
        ) as f:
            f.write(code)
            name = f.name
        return Path(name)

    def _check_code_type(self, code_type: str) -> str:
        if code_type not in self._CODE_TYPE_MAPPING:
            raise ValueError(
                f"Unsupported code type {code_type}. Currently "
                f"`{self.__class__.__name__}` only supports "
                f"{', '.join(self._CODE_EXTENSION_MAPPING.keys())}."
            )
        return self._CODE_TYPE_MAPPING[code_type]

    def supported_code_types(self) -> List[str]:
        r"""Provides supported code types by the interpreter."""
        return list(self._CODE_EXTENSION_MAPPING.keys())
