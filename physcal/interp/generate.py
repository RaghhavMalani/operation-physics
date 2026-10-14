"""Fit-set completions for the Phase 9.5 probe: SFT-only greedy on calibration-split seeds 20,000,400.. (GPU).

    uv run python -m physcal.interp.generate
"""

from physcal.config import REPO_ROOT, load_config
from physcal.eval.posthoc import generate_sft_completions


def main() -> None:
    cfg = load_config("probe_sft")
    generate_sft_completions(cfg["label"], cfg["budget"], cfg["calibration_ext_split"],
                             REPO_ROOT / cfg["output_dir"] / "sft_calibration_ext", "probe-sft-calibration-ext")


if __name__ == "__main__":
    main()
