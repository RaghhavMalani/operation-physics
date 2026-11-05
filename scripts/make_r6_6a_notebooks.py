#!/usr/bin/env python
"""D175: writes notebooks/kaggle_r6_6a.ipynb and notebooks/kaggle_r6_6a_smoke.ipynb, pinned to a pushed commit.

    uv run python scripts/make_r6_6a_notebooks.py --commit <40-hex>

The cells are thin: `python -m physcal.eval.r6_6a run` does the work, so a change of logic is a change of the pinned code,
which tests/test_r6_6a.py's pin test catches.
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POOLS = ("r6_6a_pool.jsonl", "r6_6a_pool_manifest.json", "r6_6a_smoke_pool.jsonl", "r6_6a_smoke_pool_manifest.json")

INTRO = {
    "full": """# R6-6a on Kaggle "GPU T4 x2" (D175; exploratory, paper #2, not paper #1)

Probe-to-verbal-confidence distillation: arms A to E on 2,400 fresh problems. The rules are in
`docs/exploratory/DECISIONS_FOLLOWUP.md` D175 and D171, and the steps in `docs/exploratory/kaggle_instructions.md` §9.

The notebook clones the repo at `COMMIT`, installs one venv, downloads the pinned weights, and runs every stage: sampling
(vLLM), reading, the teachers, 10 LoRA runs, test scoring, the analysis. Work is saved in chunks to `/kaggle/working`, it
stops itself by 9.5 h after the start, and it writes `r6_6a_results.zip`.

**Settings:** Accelerator **GPU T4 x2**, Internet **On**. **Run with** Save Version → **Save & Run All (Commit)**.
**Do not edit the cells.** To resume, attach this notebook's latest version as an input and run it again.""",
    "smoke": """# R6-6a's smoke session on Kaggle "GPU T4 x2" (D175; ~45 minutes of quota in all)

Every stage's real code on 48 problems of its own, to answer three questions before R6-6a: **does output survive a crash,
what is the machine, and what are the real rates (fp16 HF + PEFT LoRA on a T4)?** Its numbers are plumbing, never results.
The steps are in `docs/exploratory/kaggle_instructions.md` §9.

- **Version 1 (~35 min):** runs everything, writes `r6_6a_smoke_results.zip`, then **raises an exception on purpose**. The
  version will show **Error**. That is the test: are its files still in the Output tab?
- **Version 2 (~8 min):** with version 1's output attached as an input, it checks what came back, re-runs (every stage must
  be skipped), then **kills its own kernel on purpose**. Again, check the Output tab.

**Settings:** Accelerator **GPU T4 x2**, Internet **On**. **Run with** Save Version → **Save & Run All (Commit)**.
**Do not edit the cells.**""",
}

SETTINGS = '''# Settings and the machine. Nothing to edit: COMMIT pins the code (D171).
import json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

SESSION_START = time.time()                      # D171: no chunk may end after the session's deadline
COMMIT = "{commit}"
MODE = "{mode}"
NAME = "{name}"
REPO_URL = "https://github.com/RaghhavMalani/operation-physics.git"
WORK = Path("/kaggle/working")                   # saved as the version's output
TMP = Path("/tmp/physcal")                       # repo, venv, weights: not saved
REPO, VENV, MODEL_DIR = TMP / "repo", TMP / "venv", TMP / "qwen2.5-math-1.5b-instruct"
TMP.mkdir(parents=True, exist_ok=True)
(WORK / NAME / "logs").mkdir(parents=True, exist_ok=True)
LOG = WORK / NAME / "logs" / f"notebook_{{int(SESSION_START)}}.log"


def sh(cmd, tail=4000, **kw):
    """Run a shell command; its output goes to the cell (the last `tail` characters) and to the session log.
    A failure raises, which stops Save & Run All before anything runs."""
    print(f"$ {{cmd}}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    out = r.stdout + r.stderr
    print(out[-tail:], flush=True)
    with LOG.open("a") as fh:
        fh.write(f"$ {{cmd}}\\n{{out}}\\n")
    if r.returncode:
        raise RuntimeError(f"exit {{r.returncode}}: {{cmd}}")
    return r.stdout


GPUS = [line.split(", ") for line in
        sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader").strip().splitlines()]
assert len(GPUS) == 2 and all("T4" in g[0] for g in GPUS), f"needs Accelerator 'GPU T4 x2', got {{GPUS}}"
sh("nproc; free -g; df -h /tmp /kaggle/working")'''

REPO_CELL = '''# The repo at COMMIT (sparse: the code, the configs and the frozen pools).
if not (REPO / ".git").exists():
    sh(f"git init -q {{REPO}}")
    sh(f"git -C {{REPO}} remote add origin {{REPO_URL}}")
    sh(f"git -C {{REPO}} sparse-checkout set --no-cone /physcal/ /configs/ {pools}")
    sh(f"git -C {{REPO}} fetch -q --depth 1 --filter=blob:none origin {{COMMIT}}")
    sh(f"git -C {{REPO}} checkout -q FETCH_HEAD")
HEAD = sh(f"git -C {{REPO}} rev-parse HEAD").strip()
assert HEAD == COMMIT, (HEAD, COMMIT)
try:
    import yaml
except ImportError:
    sh(f"{{sys.executable}} -m pip install -q pyyaml==6.0.3")
    import yaml
RC = yaml.safe_load((REPO / "configs" / "r6_6a.yaml").read_text())
KC = yaml.safe_load((REPO / "configs" / "followup.yaml").read_text())["scale"]["kaggle"]
DRIVER = min(int(g[1].split(".")[0]) for g in GPUS)
assert DRIVER >= KC["min_driver"], f"driver {{DRIVER}} < {{KC['min_driver']}}: the pinned vLLM needs CUDA 13"
print(f"repo {{HEAD}} | mode {{MODE}} | driver {{DRIVER}} | {{shutil.disk_usage('/tmp').free / 1e9:.0f}} GB free on /tmp", flush=True)'''

VENV_CELL = '''# The venv (uv; isolated from Kaggle's preinstalled packages), pinned in the config.
os.environ["UV_CACHE_DIR"] = str(TMP / "uv-cache")
os.environ["UV_PYTHON_INSTALL_DIR"] = str(TMP / "uv-python")
sh(f"{sys.executable} -m pip install -q uv=={KC['uv']}")
UV = f"{sys.executable} -m uv"
if not (VENV / "bin" / "python").exists():
    sh(f"{UV} venv -q --python {KC['python']} {VENV}")
sh(f"{UV} pip install -q --python {VENV}/bin/python " + " ".join(RC["packages"]))
PY = str(VENV / "bin" / "python")
ENV = {**os.environ, "PYTHONPATH": str(REPO), "HF_HOME": str(TMP / "hf"), "TOKENIZERS_PARALLELISM": "false",
       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_USE_FLASHINFER_SAMPLER": "0"}  # no FlashInfer kernels on sm75
sh(f"{PY} -c \\"import torch, vllm, peft, sklearn, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, "
   f"'gpus', torch.cuda.device_count(), 'vllm', vllm.__version__, 'peft', peft.__version__, 'transformers', "
   f"transformers.__version__, 'sklearn', sklearn.__version__)\\"", env=ENV)'''

RESTORE_CELL = '''# A previous version's output attached as an input (a resume), then the weights at the pinned revision (sha256-checked).
def m(args):
    return sh(f"{PY} -m physcal.eval.r6_6a {args} --mode {MODE}", env=ENV, cwd=str(TMP))


m(f"restore --out {WORK} --src /kaggle/input")
if not (MODEL_DIR / "_verified.json").exists():
    m(f"download --model-dir {MODEL_DIR}")
print((MODEL_DIR / "_verified.json").read_text())'''

RUN_CELL = '''# Every stage, in order. Lines stream here and into the run's logs. This cell never raises, so the zip below is written.
cmd = [PY, "-m", "physcal.eval.r6_6a", "run", "--mode", MODE, "--out", str(WORK), "--model-dir", str(MODEL_DIR),
       "--session-start", repr(SESSION_START)]
print("$ " + " ".join(cmd), flush=True)
try:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV, cwd=str(TMP), bufsize=1)
    with LOG.open("a") as fh:
        for line in p.stdout:
            print(line, end="", flush=True)
            fh.write(line)
            fh.flush()
    RUN_EXIT = p.wait()
except Exception as exc:
    print(f"the run could not continue: {exc!r}", flush=True)
    RUN_EXIT = -1
print(f"run: exit {RUN_EXIT}", flush=True)'''

PACK_CELL = '''# The results zip, and the status line.
m(f"pack --out {WORK}")
st = WORK / NAME / "status.json"
STATUS = json.loads(st.read_text()) if st.exists() else {"stopped_at": "before the first stage"}
print("STATUS:", json.dumps({"mode": MODE, "run_exit": RUN_EXIT, "stopped_at": STATUS.get("stopped_at"), "done": STATUS.get("done")}), flush=True)
print("next:", "download the zip from the Output tab" if STATUS.get("stopped_at") is None and RUN_EXIT == 0
      else "attach this version's output as an input and run again (kaggle_instructions.md §9)", flush=True)'''

CRASH_CELL = '''# The deliberate crash (D175): the question is whether /kaggle/working survives it. Version 1 raises; version 2, which found
# version 1's output, kills its own kernel. Both versions show an error on purpose: check the Output tab.
print(f"output so far: {sum(1 for f in WORK.rglob('*') if f.is_file())} files, "
      f"{sum(f.stat().st_size for f in WORK.rglob('*') if f.is_file()) / 1e6:.0f} MB", flush=True)
if (WORK / NAME / "smoke" / "survival.json").exists():
    print("version 2: a hard kill of this kernel now", flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
raise RuntimeError("version 1: the deliberate crash of D175's smoke session; this error is expected")'''


def cell(kind: str, text: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    return c | ({"execution_count": None, "outputs": []} if kind == "code" else {})


def notebook(commit: str, mode: str) -> dict:
    name = "r6_6a" if mode == "full" else "r6_6a_smoke"
    pools = " ".join(f"/data/eval_sets/1.1.0+c735a0d4/{p}" for p in POOLS)
    cells = [cell("markdown", INTRO[mode]), cell("code", SETTINGS.format(commit=commit, mode=mode, name=name)),
             cell("code", REPO_CELL.format(pools=pools)), cell("code", VENV_CELL), cell("code", RESTORE_CELL),
             cell("code", RUN_CELL), cell("code", PACK_CELL)]
    if mode == "smoke":
        cells.append(cell("code", CRASH_CELL))
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True,
                                    "language": "python", "sourceType": "notebook"}}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    for mode, f in (("full", "kaggle_r6_6a.ipynb"), ("smoke", "kaggle_r6_6a_smoke.ipynb")):
        (REPO / "notebooks" / f).write_text(json.dumps(notebook(a.commit, mode), indent=1) + "\n")


if __name__ == "__main__":
    main()
