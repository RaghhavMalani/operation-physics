"""Phase 9.5 step 3: split credit (σ tokens get r_sigma's advantage, every other token r_answer's)."""

import copy

import pytest
import torch

from physcal.config import REPO_ROOT, load_config
from physcal.train.split_credit import (
    group_advantages,
    sigma_char_span,
    sigma_token_mask,
    split_from_breakdown,
    token_advantages,
)
from physcal.verify.reward import reward, reward_config

CFG = reward_config()
PROBLEM = {
    "target_symbol": "v", "target_value_si": 12.4, "target_si_unit": "m/s", "target_unit": "m/s",
    "intermediates": {"a1": 2.45}, "intermediate_si_units": {"a1": "m/s^2"},
    "givens": [{"symbol": "u", "desc": "", "display": "2.36", "unit": "m/s", "value_si": 2.36}], "distractors": [],
}
MODEL = REPO_ROOT / "results" / "sft" / load_config("probe_sft")["label"] / "merged"


def done(answer: str) -> str:
    return f"work...\n<steps>\na1 = 2.45 m/s^2\n</steps>\n<answer>{answer}</answer>"


@pytest.mark.parametrize("answer, sigma", [("12.5 ± 0.3 m/s", "0.3"), ("12.5 +/- 0.03 m/s", "0.03"),
                                            ("(1.25 ± 0.02) × 10^1 m/s", "0.02"), ("12.5 m/s ± 0 m/s", "0"),
                                            ("12.5 \\pm 1.5e-2 m/s", "1.5e-2"), ("12.5 ± 2% m/s", "2")])
def test_sigma_char_span_finds_the_sigma_number(answer, sigma):
    text = done(answer)
    a, b = sigma_char_span(text)
    assert text[a:b] == sigma


def test_sigma_char_span_none_without_single_answer_or_pm():
    assert sigma_char_span(done("12.5 m/s")) is None
    assert sigma_char_span("<answer>1 ± 1 m</answer><answer>2 ± 1 m</answer>") is None


def test_parts_add_up_to_the_verifier_reward_and_vanish_at_sigma_ref():
    for answer in ("12.5 ± 0.3 m/s", "12.4 ± 0 m/s", "20 ± 1 m/s", "12.41 ± 0.0001 m/s"):
        r, bd = reward(PROBLEM, done(answer), CFG)
        p = split_from_breakdown(r, bd, done(answer), CFG, 0.01)
        assert p["r_answer"] + p["r_sigma"] == pytest.approx(r, abs=1e-12)
    r, bd = reward(PROBLEM, done("12.5 ± 0.125 m/s"), CFG)  # σ = 1% of μ = σ_ref
    p = split_from_breakdown(r, bd, done("12.5 ± 0.125 m/s"), CFG, 0.01)
    assert p["r_sigma"] == pytest.approx(0.0, abs=1e-12) and p["r_answer"] == pytest.approx(r)


def test_failures_keep_everything_in_r_answer():
    for text in ("no answer at all", done("12.5 ± 0.3 kg")):
        r, bd = reward(PROBLEM, text, CFG)
        p = split_from_breakdown(r, bd, text, CFG, 0.01)
        assert p == {"r_total": r, "r_answer": r, "r_sigma": 0.0, "span": None}


def test_group_advantages_match_trl_formula():
    r = torch.tensor([1.0, 2.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    a = group_advantages(r, 4)
    g = r.view(2, 4)
    expected = ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-4)).view(-1)
    assert torch.allclose(a, expected) and torch.all(a[4:] == 0)


# ------------------------------------------------------------------------- gradient properties of the loss
def dapo_loss(new: torch.Tensor, old: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor, eps: float = 0.2):
    """The per-token GRPO/DAPO objective TRL and Unsloth compute (token-level importance ratio, β = 0)."""
    adv = adv.unsqueeze(1) if adv.dim() == 1 else adv
    coef_1 = torch.exp(new - old)
    coef_2 = torch.clamp(coef_1, 1 - eps, 1 + eps)
    loss_i = -torch.min(coef_1 * adv, coef_2 * adv)
    return (loss_i * mask).sum() / mask.sum()


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    B, T = 8, 7
    new = torch.randn(B, T, generator=g) * 0.1
    old = new.detach() + torch.randn(B, T, generator=g) * 0.05  # a mild off-policy ratio, so clipping is exercised
    mask = torch.ones(B, T)
    mask[:, 6] = 0  # padding
    smask = torch.zeros(B, T, dtype=torch.bool)
    smask[:6, 4:6] = True  # rows 6-7 have no σ tokens (e.g. parse failures)
    return new, old, mask, smask


def _grads(new, old, mask, smask, r_ans, r_sig):
    adv = token_advantages(group_advantages(r_ans, 4), group_advantages(r_sig, 4), smask)
    x = new.clone().requires_grad_(True)
    dapo_loss(x, old, adv, mask).backward()
    return x.grad


def test_sigma_token_gradients_depend_only_on_r_sigma_and_answer_tokens_only_on_r_answer():
    new, old, mask, smask = _batch()
    r_ans = torch.tensor([0.1, -0.3, 0.5, 0.0, -1.0, -0.2, 0.3, 0.9])
    r_sig = torch.tensor([0.01, -0.02, 0.0, 0.03, 0.0, 0.002, 0.0, 0.0])
    base = _grads(new, old, mask, smask, r_ans, r_sig)
    other_ans = _grads(new, old, mask, smask, r_ans * -2 + 1, r_sig)
    other_sig = _grads(new, old, mask, smask, r_ans, torch.flip(r_sig, [0]) * 7)
    ans_tokens = (~smask) & mask.bool()
    assert torch.equal(base[smask], other_ans[smask])            # σ tokens ignore r_answer
    assert not torch.allclose(base[ans_tokens], other_ans[ans_tokens])
    assert torch.equal(base[ans_tokens], other_sig[ans_tokens])  # answer tokens ignore r_sigma
    assert not torch.allclose(base[smask], other_sig[smask])
    assert torch.all(base[~mask.bool()] == 0)


def test_with_sigma_ref_equal_to_sigma_it_reduces_to_the_original_loss_on_answer_tokens():
    """σ_ref = σ ⇒ r_sigma = 0, r_answer = r ⇒ answer-token gradients equal the original (uniform-credit) ones."""
    new, old, mask, smask = _batch(1)
    r = torch.tensor([0.1, -0.3, 0.5, 0.0, -1.0, -0.2, 0.3, 0.9])
    split = _grads(new, old, mask, smask, r, torch.zeros(8))
    x = new.clone().requires_grad_(True)
    dapo_loss(x, old, group_advantages(r, 4), mask).backward()
    ans_tokens = (~smask) & mask.bool()
    assert torch.allclose(split[ans_tokens], x.grad[ans_tokens], atol=1e-7, rtol=0)
    assert torch.all(split[smask] == 0)


def test_reward_parts_built_from_real_completions_with_sigma_ref_equal_sigma():
    """End to end on reward parts: σ = 1% of μ for every completion gives r_sigma ≡ 0 and r_answer ≡ r."""
    texts = [done(f"{v} ± {round(v * 0.01, 6)} m/s") for v in (12.4, 12.5, 13.1, 9.0)]
    parts = [split_from_breakdown(*reward(PROBLEM, t, CFG), t, CFG, 0.01) for t in texts]
    assert all(abs(p["r_sigma"]) < 1e-12 for p in parts)
    assert torch.allclose(group_advantages(torch.tensor([p["r_answer"] for p in parts]), 4),
                          group_advantages(torch.tensor([p["r_total"] for p in parts]), 4))


@pytest.mark.skipif(not MODEL.exists(), reason="local SFT model not present")
def test_sigma_token_mask_marks_exactly_the_sigma_number_tokens():
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(str(MODEL))
    texts = [done("12.5 ± 0.25 m/s"), done("1.234e3 ± 12.5 kg"), "no answer", done("7 m")]
    enc = [tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id] for t in texts]
    T = max(map(len, enc)) + 2
    ids = torch.full((4, T), tok.pad_token_id)
    cmask = torch.zeros(4, T, dtype=torch.long)
    for b, e in enumerate(enc):
        ids[b, :len(e)] = torch.tensor(e)
        cmask[b, :len(e)] = 1
    smask, unmatched = sigma_token_mask(tok, ids, cmask, texts, [sigma_char_span(t) for t in texts])
    assert unmatched == 0
    assert tok.decode(ids[0][smask[0]]).strip() == "0.25"
    assert tok.decode(ids[1][smask[1]]).strip() == "12.5"
    assert not smask[2].any() and not smask[3].any()


def test_split_credit_config_differs_from_the_control_only_in_split_credit():
    old, new = load_config("grpo_overnight2"), load_config("grpo_split_credit")
    new = copy.deepcopy(new)
    assert new.pop("split_credit") == {"sigma_ref_rel": 0.01}
    assert new.pop("run_id") != old.pop("run_id")
    assert new == old


UNSLOTH_CHECK = r"""
import sys, torch
import unsloth  # noqa: F401  (patches trl's GRPOTrainer)
from trl import GRPOTrainer
loss_fn = sys.modules[GRPOTrainer.__module__].grpo_compute_loss_slow
sys.path.insert(0, "tests")
from test_split_credit import _batch, dapo_loss
from physcal.train.split_credit import group_advantages, token_advantages
new, old, mask, smask = _batch(2)
adv = token_advantages(group_advantages(torch.randn(8), 4), group_advantages(torch.randn(8), 4), smask)
x1 = new.clone().requires_grad_(True); x2 = new.clone().requires_grad_(True)
ids = torch.zeros(8, 7, dtype=torch.long)
out = loss_fn(None, x1, old, None, ids, mask, 0.0, adv, loss_type="dapo", num_items_in_batch=mask.sum(),
              current_gradient_accumulation_steps=1, steps_per_generation=1, num_processes=1)
ref = dapo_loss(x2, old, adv, mask)
out[0].backward(); ref.backward()
assert torch.allclose(out[0], ref, atol=1e-6), (out[0], ref)
assert torch.allclose(x1.grad, x2.grad, atol=1e-7)

# pack_like_unsloth must put each completion token's advantage exactly where Unsloth's fused loss puts that token.
m = sys.modules[GRPOTrainer.__module__]
from physcal.train.split_credit import pack_like_unsloth
PAD = 0
prompt = torch.tensor([[PAD, PAD, 11, 12], [PAD, 21, 22, 23], [31, 32, 33, 34]])
comp = torch.tensor([[101, 102, 103, PAD, PAD], [201, 202, 203, 204, 205], [301, PAD, PAD, PAD, PAD]])
L = comp.size(1)
ids = torch.cat([prompt, comp], 1)
lp = m.calculate_pad_tokens_in_prompt(ids, L, PAD)
M = int(lp.max())
window = m.left_pack_padding(ids, PAD)[:, -(L + M):]
cmask = m.create_completion_attention_mask(window, lp, M, PAD)
packed = pack_like_unsloth(comp.float(), prompt, comp, PAD)  # advantage := token id, to track positions
assert packed.shape == window.shape
assert torch.equal(packed[cmask], window[cmask].float()), (packed, window, cmask)
print("OK")
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="unsloth needs a GPU to import")
def test_unsloth_loss_accepts_per_token_advantages_and_matches_the_reference():
    """The real Unsloth GRPO loss, fed (B, T) advantages, gives the same loss and gradients as `dapo_loss`."""
    import subprocess
    import sys

    res = subprocess.run([sys.executable, "-c", UNSLOTH_CHECK], cwd=REPO_ROOT, capture_output=True, text=True,
                         timeout=600)
    assert res.returncode == 0 and res.stdout.strip().endswith("OK"), res.stderr[-2000:]


def test_pack_like_unsloth_shifts_rows_by_their_prompt_padding():
    from physcal.train.split_credit import pack_like_unsloth

    prompt = torch.tensor([[0, 0, 5], [0, 5, 5], [5, 5, 5]])  # left pads 2, 1, 0 -> M = 2
    comp = torch.tensor([[7, 8], [7, 8], [7, 8]])
    adv = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = pack_like_unsloth(adv, prompt, comp, 0)
    assert out.tolist() == [[1.0, 2.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0], [0.0, 0.0, 5.0, 6.0]]
