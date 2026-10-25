"""Internal isolated worker. The only input/output channel is JSON on stdio."""
import builtins
import json
from pathlib import Path
import resource
import sys

# -I -S intentionally excludes the repository and site packages from sys.path.
# Load just the validator module; never import Habitat, model clients or .env.
import importlib.util
spec = importlib.util.spec_from_file_location("recovery_program", Path(__file__).with_name("program.py"))
program = importlib.util.module_from_spec(spec)
spec.loader.exec_module(program)


def main():
    request = json.loads(sys.stdin.read(2_000_001))
    resource.setrlimit(resource.RLIMIT_CPU, (request["cpu_seconds"], request["cpu_seconds"]))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    tree = program.validate_code(request["code"])
    namespace = {"__builtins__": {name: getattr(builtins, name) for name in program.BUILTINS}}
    exec(compile(tree, "<generated-recovery>", "exec"), namespace)
    value = program.validate_result(namespace["recover"](request["context"], request["memory"]))
    output = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(output.encode("utf-8")) > 262144:
        raise ValueError("Recovery result exceeds 256KB")
    sys.stdout.write(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.stderr.write(type(error).__name__ + ": " + str(error)[:1000])
        sys.exit(1)
