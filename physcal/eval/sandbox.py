"""MC-B's program checks and sandbox (DECISIONS.md D119 §4): a model-written `solve(...)` is extracted from its
completion, checked statically (AST), then run in a separate `python -I` process with resource limits.

Static check: imports only `math`; no name, attribute, parameter or function whose name starts with `_`; no call or
reference to the forbidden builtins (eval, exec, open, compile, getattr, setattr, delattr, globals, locals, vars,
input, breakpoint, exit, quit, __import__); no class, global, nonlocal, with or async anywhere, and no lambda at
module level. The signature: positional-or-keyword parameters only; each one is a displayed given's symbol, or has a
default and is then not passed (e.g. `g=9.81`).

The child (`sandbox_child.py`) sets RLIMIT_CPU / RLIMIT_AS / RLIMIT_FSIZE before reading the program, execs it with
restricted builtins (`math` injected; `import math` is the only import that resolves; `print` is a sink) and calls
`solve(**kwargs)` once per input set, all in the same process. The parent adds a wall timeout.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

CHILD = Path(__file__).with_name("sandbox_child.py")
FENCE_RE = re.compile(r"```[ \t]*python[ \t]*\n(.*?)```", re.S | re.I)
FORBIDDEN_NAMES = frozenset({"eval", "exec", "open", "compile", "getattr", "setattr", "delattr", "globals", "locals",
                             "vars", "input", "breakpoint", "exit", "quit", "__import__"})
FORBIDDEN_NODES = (ast.ClassDef, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.AsyncFunctionDef, ast.AsyncFor,
                   ast.Await)
FAILURES = ("no_program", "rejected", "signature", "exception", "timeout", "bad_output")


def extract(completion: str) -> str | None:
    """The first fenced ```python block that defines `solve`."""
    for block in FENCE_RE.findall(completion):
        if re.search(r"^\s*def\s+solve\s*\(", block, re.M):
            return block
    return None


def static_check(code: str) -> tuple[ast.Module | None, str | None]:
    """(the parsed module, None) if the program passes, else (None, the reason)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return None, f"syntax: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            return None, f"node {type(node).__name__}"
        if isinstance(node, ast.Import) and any(a.name != "math" for a in node.names):
            return None, "import other than math"
        if isinstance(node, ast.ImportFrom) and (node.module != "math" or node.level):
            return None, "import other than math"
        names = []
        if isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, (ast.FunctionDef, ast.arg)):
            names = [node.name if isinstance(node, ast.FunctionDef) else node.arg]
        elif isinstance(node, ast.alias):
            names = [node.asname or node.name]
        for n in names:
            if n.startswith("_"):
                return None, f"underscore name {n}"
            if n in FORBIDDEN_NAMES:
                return None, f"forbidden name {n}"
    for stmt in tree.body:
        if not isinstance(stmt, ast.FunctionDef) and any(isinstance(n, ast.Lambda) for n in ast.walk(stmt)):
            return None, "lambda at module level"
    return tree, None


def signature(tree: ast.Module, displayed: set[str]) -> tuple[list[str] | None, str | None]:
    """The parameters to pass (each a displayed symbol), or the reason the signature fails."""
    defs = [s for s in tree.body if isinstance(s, ast.FunctionDef) and s.name == "solve"]
    if not defs:
        return None, "no top-level solve"
    a = defs[-1].args  # the last definition is the one that runs
    if a.posonlyargs or a.vararg or a.kwarg or a.kwonlyargs:
        return None, "not positional-or-keyword parameters only"
    n_default = len(a.defaults)
    params = []
    for i, arg in enumerate(a.args):
        has_default = i >= len(a.args) - n_default
        if arg.arg in displayed:
            params.append(arg.arg)
        elif not has_default:
            return None, f"parameter {arg.arg} is not a displayed given"
    return params, None


def run_program(code: str, calls: list[dict[str, float]], cpu_s: int, mem_bytes: int, wall_s: float) -> dict:
    """Run `solve(**kw)` for every kw in `calls` in one sandboxed child. Returns {"ok": bool, "failure": str | None,
    "values": [float | None per call], "errors": [...]}; any failed call fails the program."""
    payload = json.dumps({"code": code, "calls": calls, "cpu_s": cpu_s, "mem_bytes": mem_bytes})
    try:
        r = subprocess.run([sys.executable, "-I", str(CHILD)], input=payload, capture_output=True, text=True,
                           timeout=wall_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "failure": "timeout", "values": [None] * len(calls), "errors": ["wall timeout"]}
    if r.returncode != 0 or not r.stdout.strip():
        failure = "timeout" if r.returncode in (-24, -9) else "exception"  # SIGXCPU / SIGKILL at the CPU limit
        return {"ok": False, "failure": failure, "values": [None] * len(calls),
                "errors": [f"child exit {r.returncode}: {r.stderr[-300:]}"]}
    out = json.loads(r.stdout.strip().splitlines()[-1])
    if out.get("module_error"):
        return {"ok": False, "failure": "exception", "values": [None] * len(calls), "errors": [out["module_error"]]}
    values, errors, failure = [], [], None
    for res in out["results"]:
        if res.get("error"):
            failure = failure or "exception"
            values.append(None)
            errors.append(res["error"])
        elif res.get("bad_output"):
            failure = failure or "bad_output"
            values.append(None)
            errors.append(res["bad_output"])
        else:
            values.append(res["value"])
            errors.append(None)
    return {"ok": failure is None, "failure": failure, "values": values, "errors": errors}
