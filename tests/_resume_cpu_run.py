"""Helper for tests/test_resume.py (D66), run in a subprocess with CUDA hidden: plain TRL GRPOTrainer (no Unsloth)
on a tiny random Qwen2 + LoRA, wired by the real `physcal.train.grpo.setup_trainer`.

    python tests/_resume_cpu_run.py <tiny model dir> <run dir> <steps> full
    python tests/_resume_cpu_run.py <tiny model dir> <run dir> <steps> resume <checkpoint dir, relative to run dir>
    python tests/_resume_cpu_run.py <tiny model dir> <run dir> <steps> naive <absolute checkpoint dir>

`naive` is the negative control (D63's failure): a fresh run dir and curriculum, and the trainer's own resume with its
default data skipping.

The model's completions are noise, so the binary outcome the curriculum observes is replaced by a fixed function of
the problem seed: the curriculum's path then depends only on the sampler state and the order of draws and
observations, which is what resuming has to preserve.
"""

import copy
import sys
from pathlib import Path

from physcal.config import load_config
from physcal.train import grpo

CORRECT_EVERY = 3  # a problem counts as answered correctly iff seed % CORRECT_EVERY == 0


def fake_rule_outcomes(problem, text, cfgs):
    return {"binary": {"score_final": 1.0 if problem.seed % CORRECT_EVERY == 0 else 0.0}}


def cpu_config(steps: int) -> dict:
    """The split-credit run's config, shrunk for CPU, with a fast-moving curriculum (so that a wrongly restored
    sampler changes which cells are drawn). Split credit itself is dropped: its loss expects Unsloth's packed
    layout, and it keeps no state across steps."""
    cfg = copy.deepcopy(load_config("grpo_split_credit"))
    cfg.pop("split_credit")
    cfg["prompt"] = "v1"
    cfg["curriculum"].update(pool_per_cell=4, alpha=0.5, min_observations=1)  # weights and unlocks move fast
    cfg["grpo"].update(max_steps=steps, per_device_train_batch_size=4, gradient_accumulation_steps=2,
                       num_generations=4, max_completion_length=8, max_prompt_length=None, save_steps=2,
                       save_total_limit=None, permanent_save_steps=[])
    return cfg


def main(model_dir: str, out: str, steps: str, mode: str, ckpt: str | None = None) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    grpo.rule_outcomes = fake_rule_outcomes
    cfg = cpu_config(int(steps))
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32),
                           LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    st = grpo.setup_trainer(cfg, "cpu-resume-check", Path(out), model, tok, int(steps), None,
                            ckpt if mode == "resume" else None, GRPOTrainer, GRPOConfig,
                            extra_args={"use_cpu": True, "bf16": False, "optim": "adamw_torch"}, memory_log=False)
    trainer = st["trainer"]
    if mode == "naive":
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train(resume_from_checkpoint=str(st["resume_ckpt"])) if st["resume_ckpt"] else trainer.train()
    st["tape"].close()


if __name__ == "__main__":
    main(*sys.argv[1:])
