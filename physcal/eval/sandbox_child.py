"""The sandbox child for MC-B (DECISIONS.md D119 §4), run as `python -I sandbox_child.py` with the request on stdin:
{"code": str, "calls": [{param: value}], "cpu_s": int, "mem_bytes": int}. Limits are set before the program is
read. Writes one JSON line: {"results": [{"value": float} | {"error": str} | {"bad_output": str}]} or
{"module_error": str}. Only the standard library; nothing from the repo is importable here (-I)."""

import io
import json
import math
import resource
import sys


def _limits(cpu_s: int, mem_bytes: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if name != "math" or level:
        raise ImportError(f"import of {name!r} is not allowed")
    return math


SAFE_BUILTINS = {n: __builtins__.__dict__[n] if hasattr(__builtins__, "__dict__") else __builtins__[n] for n in (
    "abs", "min", "max", "sum", "round", "pow", "range", "len", "float", "int", "bool", "list", "tuple", "dict", "set",
    "enumerate", "zip", "sorted", "reversed", "map", "filter", "any", "all", "isinstance", "divmod", "complex",
    "ValueError", "ZeroDivisionError", "ArithmeticError", "OverflowError", "TypeError", "Exception")}


def main() -> None:
    req = json.loads(sys.stdin.read())
    _limits(int(req["cpu_s"]), int(req["mem_bytes"]))
    stdout, sink = sys.stdout, io.StringIO()
    env = {"__builtins__": {**SAFE_BUILTINS, "__import__": _import, "print": lambda *a, **k: None},
           "math": math, "__name__": "solution"}
    sys.stdout = sink
    try:
        exec(compile(req["code"], "<solution>", "exec"), env)
        solve = env["solve"]
    except BaseException as exc:  # noqa: BLE001 - every failure is reported, never raised
        sys.stdout = stdout
        print(json.dumps({"module_error": f"{type(exc).__name__}: {exc}"[:300]}))
        return
    results = []
    for kw in req["calls"]:
        try:
            y = solve(**kw)
        except BaseException as exc:  # noqa: BLE001
            results.append({"error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        try:
            v = float(y)
        except (TypeError, ValueError) as exc:
            results.append({"bad_output": f"{type(y).__name__}: {exc}"[:300]})
            continue
        if not (math.isfinite(v) and v > 0):
            results.append({"bad_output": f"value {v!r}"})
        else:
            results.append({"value": v})
    sys.stdout = stdout
    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
