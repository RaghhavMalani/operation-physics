"""R5-6 part J (DECISIONS.md D147): the training support of the Jacobians, from the R4-1 problems.

D-pos-v2 and B-v2 are SFT-only on the same 3,000 R4-1 problems (D113 §1), so these are every problem either model was
trained on. The support file stores, per chain (family + exact step sequence), every training problem's
log-sensitivities d ln f / d ln x at its displayed givens (what the builder's S lines show), for every chain given;
per quantity (family + symbol), the range of its SI value over the training problems (givens, intermediates, target);
and per (family, given, target), the range of the given's sensitivity over every training chain (D147 §10's strict
subgroup). Every J pair stores the file's SHA-256 in its construction, and replay refuses another file.

    nice uv run python -m physcal.env.jacobian_support build   # data/jacobian_support/r4_1.json
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache

from physcal.config import REPO_ROOT, load_config
from physcal.env import noise as noise_mode
from physcal.env.formulas import get_family
from physcal.env.graph import Chain, Family


def chain_key(family: str, steps) -> str:
    return family + "|" + ";".join(f"{f}:{o}" for f, o in steps)


def _sha256(path) -> str:
    return hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()


def _chain_of(fam: Family, steps) -> Chain:
    key = tuple(tuple(s) for s in steps)
    for cs in fam.chains(len(key)).values():
        for c in cs:
            if c.steps == key:
                return c
    raise KeyError(f"{fam.name}: no chain {key}")


def build(cfg: dict | None = None) -> dict:
    """The support of the configured training problems (pairs.yaml `r5_6.support`)."""
    cfg = cfg or load_config("pairs")["r5_6"]["support"]
    step = load_config(load_config("pairs")["env_config"])["generator"]["condition_rel_step"]
    chains: dict[str, dict] = {}
    mags: dict[str, list[float]] = {}
    gt: dict[str, list[float]] = {}
    lines = (REPO_ROOT / cfg["training_problems"]).read_text().splitlines()
    for line in lines:
        r = json.loads(line)
        fam = get_family(r["family"])
        chain = _chain_of(fam, r["chain"])
        shown = {g["symbol"]: g["value_si"] for g in r["givens"]}
        sens = noise_mode.log_sensitivities(fam, chain, shown, step)
        c = chains.setdefault(chain_key(r["family"], chain.steps), {"givens": list(chain.givens), "rows": []})
        c["rows"].append([sens[g] for g in chain.givens])
        for g in chain.givens:
            gt.setdefault(f"{r['family']}|{g}|{chain.target}", []).append(sens[g])
        values = {**shown, **r["intermediates"], r["target_symbol"]: r["target_value_si"]}
        for q, v in values.items():
            mags.setdefault(f"{r['family']}|{q}", []).append(v)
    span = lambda vs: [min(vs), max(vs)]
    for c in chains.values():
        c["n"] = len(c["rows"])
        c["range"] = {g: span([row[i] for row in c["rows"]]) for i, g in enumerate(c["givens"])}
    return {"decision": "D147 §1 (R5-6 part J): the training support of the Jacobians",
            "training_problems": cfg["training_problems"], "training_problems_sha256": _sha256(cfg["training_problems"]),
            "n_problems": len(lines), "condition_rel_step": step,
            "chains": dict(sorted(chains.items())), "magnitudes": {k: span(v) for k, v in sorted(mags.items())},
            "given_target": {k: span(v) for k, v in sorted(gt.items())}}


def write(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config("pairs")["r5_6"]["support"]
    sup = build(cfg)
    out = REPO_ROOT / cfg["file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sup, indent=1, sort_keys=True) + "\n")
    return {"file": cfg["file"], "sha256": _sha256(cfg["file"]), "n_problems": sup["n_problems"],
            "n_chains": len(sup["chains"])}


@lru_cache(maxsize=None)
def _load(path: str, sha256: str) -> dict:
    got = _sha256(path)
    if got != sha256:
        raise ValueError(f"{path}: SHA-256 {got[:12]}… is not the construction's {sha256[:12]}… (D147 §5)")
    return json.loads((REPO_ROOT / path).read_text())


def load(jcfg: dict) -> dict:
    """The support a construction names (`jacobian: {support_file, support_sha256, ...}`), checksum checked."""
    return _load(jcfg["support_file"], jcfg["support_sha256"])


# ------------------------------------------------------------------------------------------- the J test and distances
def exceedance(sup: dict, family: str, chain: Chain, sens: dict[str, float]) -> tuple[float, str | None]:
    """The largest distance of a given's sensitivity outside its range for this chain, and that given; (0, None)
    inside. Raises KeyError for a chain that was never trained."""
    c = sup["chains"][chain_key(family, chain.steps)]
    best, arg = 0.0, None
    for g in chain.givens:
        lo, hi = c["range"][g]
        e = max(0.0, lo - sens[g], sens[g] - hi)
        if e > best:
            best, arg = e, g
    return best, arg


def trained(sup: dict, family: str, chain: Chain) -> bool:
    return chain_key(family, chain.steps) in sup["chains"]


def in_magnitudes(sup: dict, family: str, values: dict[str, float]) -> bool:
    """Every value inside that quantity's training range (a quantity never seen in training is out of range)."""
    for q, v in values.items():
        span = sup["magnitudes"].get(f"{family}|{q}")
        if span is None or not span[0] <= v <= span[1]:
            return False
    return True


def distances(sup: dict, family: str, chain: Chain, sens: dict[str, float], margin: float) -> dict:
    """D147 §10's per-pair report: exceedance, nearest training Jacobian (L∞), training count, the clamped
    sensitivities, the strict flag and the rank-flip flag."""
    c = sup["chains"][chain_key(family, chain.steps)]
    exc, arg = exceedance(sup, family, chain, sens)
    vec = [sens[g] for g in chain.givens]
    nn = min(max(abs(a - b) for a, b in zip(vec, row)) for row in c["rows"])
    clamped = {g: min(max(sens[g], c["range"][g][0]), c["range"][g][1]) for g in chain.givens}
    strict = []
    for g in chain.givens:
        lo, hi = sup["given_target"][f"{family}|{g}|{chain.target}"]
        if max(0.0, lo - sens[g], sens[g] - hi) >= margin:
            strict.append(g)
    top = lambda row: chain.givens[max(range(len(row)), key=lambda i: (abs(row[i]), -i))]
    return {"exceedance": exc, "exceedance_given": arg, "nearest_training_linf": nn, "n_training": c["n"],
            "out_of_range_givens": [g for g in chain.givens
                                    if max(0.0, c["range"][g][0] - sens[g], sens[g] - c["range"][g][1]) >= margin],
            "sensitivities": {g: sens[g] for g in chain.givens}, "clamped": clamped,
            "strict_givens": strict, "strict": bool(strict),
            "rank_flip": top(vec) not in {top(row) for row in c["rows"]}}


def clamped_sigma_log_ratio(dist: dict, levels: dict[str, float]) -> float:
    """ln(σ_δ / σ_δ with every sensitivity clamped into its training range), at these levels."""
    full = noise_mode.delta_sigma_log(dist["sensitivities"], levels)
    clamp = noise_mode.delta_sigma_log(dist["clamped"], levels)
    return math.log(full / clamp) if clamp > 0 else math.inf


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["build"]:
        raise SystemExit("usage: python -m physcal.env.jacobian_support build")
    print(json.dumps(write(), indent=1))
