"""Paired before/after significance on the same eval problems (greedy, so one outcome per problem).

    uv run python scripts/paired_tests.py BEFORE_TAPE AFTER_TAPE OUT_JSON

McNemar exact test (binomial on discordant pairs) for binary outcomes; Wilcoxon signed-rank for reward and for
the crps_log score alone. Coverage (phase 3.7): a problem counts as covered at 1σ / 2σ only if it was answered with
a σ and the truth lies inside the interval, so failures count as not covered (denominator = all problems).
"""

import json
import statistics as st
import sys

from scipy.stats import binomtest, wilcoxon

METRICS = ("correct_strict", "correct_sigma_free", "parsed", "truncated", "covered_1sigma", "covered_2sigma")
CRPS_LOG_FLOOR = -5.0  # failures score the rule's floor (D13)


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        p, b = r["problem"], r["breakdown"]
        out[(r["split"], p["family"], p["depth"], p["seed"])] = {
            "correct_strict": b["reason"] == "ok" and b.get("abs_rel_error", 9.0) <= 0.01,
            "correct_sigma_free": r.get("by_rule", {}).get("binary", {}).get("score_final") == 1.0,
            "parsed": b["reason"] != "parse_fail",
            "truncated": bool(r["truncated"]),
            "covered_1sigma": b["reason"] == "ok" and bool(b.get("within_1sigma")),
            "covered_2sigma": b["reason"] == "ok" and bool(b.get("within_2sigma")),
            "reward": r["reward"],
            "crps_log": (b.get("scores") or {}).get("crps_log") if b["reason"] == "ok" else CRPS_LOG_FLOOR,
        }
    return out


def main(before_path, after_path, out_path):
    before, after = load(before_path), load(after_path)
    if before.keys() != after.keys():
        raise SystemExit("before and after were not run on the same problems")
    result = {}
    for split in sorted({k[0] for k in before}):
        keys = [k for k in before if k[0] == split]
        res = {"n": len(keys)}
        for m in METRICS:
            up = sum(after[k][m] and not before[k][m] for k in keys)
            down = sum(before[k][m] and not after[k][m] for k in keys)
            res[m] = {"before": sum(before[k][m] for k in keys), "after": sum(after[k][m] for k in keys),
                      "only_after": up, "only_before": down,
                      "mcnemar_p": binomtest(up, up + down, 0.5).pvalue if up + down else 1.0}
        diffs = [after[k]["reward"] - before[k]["reward"] for k in keys]
        res["reward"] = {"mean_diff": st.mean(diffs), "nonzero": sum(d != 0 for d in diffs),
                         "wilcoxon_p": wilcoxon(diffs).pvalue if any(diffs) else 1.0}
        diffs = [after[k]["crps_log"] - before[k]["crps_log"] for k in keys]
        res["crps_log"] = {"before_mean": st.mean(before[k]["crps_log"] for k in keys),
                           "after_mean": st.mean(after[k]["crps_log"] for k in keys),
                           "mean_diff": st.mean(diffs), "nonzero": sum(d != 0 for d in diffs),
                           "wilcoxon_p": wilcoxon(diffs).pvalue if any(diffs) else 1.0}
        result[split] = res
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
