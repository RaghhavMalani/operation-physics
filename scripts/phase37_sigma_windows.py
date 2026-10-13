"""Per-25-step windows of training rollouts: share of sigma = 0 answers, median stated relative sigma, 1-sigma
coverage and accuracy (answers with sigma only).

    uv run python scripts/phase37_sigma_windows.py TAPE [TAPE ...]
"""
import json, sys, statistics as st
from collections import defaultdict
for path in sys.argv[1:]:
    w = defaultdict(list)
    for line in open(path):
        r = json.loads(line); b = r["breakdown"]
        if b["reason"] == "ok" and b.get("sigma_si") is not None:
            w[(r["step"] - 1) // 25].append((b["sigma_si"] == 0, b["rel_sigma"], b["within_1sigma"], b["abs_rel_error"] <= 0.01))
    print(path)
    for k in sorted(w):
        v = w[k]
        print(f"  steps {k*25+1:3d}-{k*25+25:3d}: n={len(v):4d} sigma0={sum(x[0] for x in v)/len(v):.2f} "
              f"med_rel_sigma={st.median(x[1] for x in v):.5f} cov1={sum(x[2] for x in v)/len(v):.2f} acc={sum(x[3] for x in v)/len(v):.2f}")
