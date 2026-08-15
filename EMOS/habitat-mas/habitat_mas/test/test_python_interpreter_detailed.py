from pathlib import Path
from typing import Optional

from habitat_mas.utils.python_interpreter import SubprocessInterpreter


class TrackingInterpreter(SubprocessInterpreter):
    last_temp_file: Optional[Path] = None

    def _create_temp_file(self, code: str, extension: str) -> Path:
        path = super()._create_temp_file(code, extension)
        self.last_temp_file = path
        return path


def test_detailed_python_execution_separates_success_output_and_cleans_up():
    interpreter = TrackingInterpreter(print_stderr=False)

    result = interpreter.run_python_detailed('print("distance: 1.25")')

    assert result.returncode == 0
    assert result.stdout == "distance: 1.25\n"
    assert result.stderr == ""
    assert result.timed_out is False
    assert interpreter.last_temp_file is not None
    assert not interpreter.last_temp_file.exists()


def test_detailed_python_execution_reports_syntax_without_launching(monkeypatch):
    interpreter = TrackingInterpreter(print_stderr=False)

    def fail_if_launched(*args, **kwargs):
        raise AssertionError("syntax-invalid source must not launch a subprocess")

    monkeypatch.setattr("habitat_mas.utils.python_interpreter.subprocess.Popen", fail_if_launched)
    result = interpreter.run_python_detailed("if True print('bad')")

    assert result.returncode != 0
    assert result.stdout == ""
    assert "SyntaxError" in result.stderr
    assert result.timed_out is False
    assert interpreter.last_temp_file is None


def test_detailed_python_execution_separates_runtime_error_and_cleans_up():
    interpreter = TrackingInterpreter(print_stderr=False)

    result = interpreter.run_python_detailed("raise ValueError('bad geometry')")

    assert result.returncode != 0
    assert result.stdout == ""
    assert "ValueError: bad geometry" in result.stderr
    assert result.timed_out is False
    assert interpreter.last_temp_file is not None
    assert not interpreter.last_temp_file.exists()


def test_detailed_python_execution_times_out_and_cleans_up():
    interpreter = TrackingInterpreter(print_stderr=False)

    result = interpreter.run_python_detailed(
        "import time\ntime.sleep(1)", timeout_seconds=0.05
    )

    assert result.returncode != 0
    assert result.timed_out is True
    assert interpreter.last_temp_file is not None
    assert not interpreter.last_temp_file.exists()


def test_legacy_run_format_is_unchanged():
    interpreter = SubprocessInterpreter(print_stderr=False)

    assert interpreter.run('print("value: 3")', "python") == "value: 3\n"
