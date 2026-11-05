"""Session-by-session simulation of R6-6a at the smoke's measured rates, with the driver's real rules: no chunk starts unless
now + 1.25 x its time <= 9.5 h. Hours. static = worker w takes keys[w::2]; dynamic = a free GPU takes the next key."""
H, M = 9.5, 1.25
S0, S1, S3 = 0.1, 3.95, 2.7         # S3: laptop fits scaled by 1.3 (7,800 answers) and 2 (the Xeon); an estimate
ANS, ID, HT, VAL, W1, SW = 18700, 3918, 6075, 937, 3918, (780, 390, 780)
def s5(runs, sec, ht_seed0=False, val_seed0=False):
    ks = []
    for r in runs:
        ks.append((r, ID))
        if not ht_seed0 or r.endswith("_0"): ks.append((r, HT))
        if not val_seed0 or r.endswith("_0"): ks.append((r, VAL))
    for m in sec:
        ks.append((m, W1)); ks += [(m, n) for n in SW]
    return ks
def sim(runs, sec, step, adapter, base, steps=375, dynamic=False, barrier=True, ht0=False, v0=False, label=""):
    """step, adapter, base: per-GPU s/step, s/item with an adapter, s/item for the base."""
    run_h = [steps * s / 3600 for s in step]
    B = [r for r in runs if r[0] == "B"]; rest = [r for r in runs if r[0] != "B"]
    train_todo = list(runs); score = s5(runs, sec, ht0, v0)
    def item_h(g, m, n): return n * (base[g] if m == "A" else adapter[g]) / 3600
    state = dict(s1=S1, s2=ANS * max(.3675, .3675) / 2 / 3600, s3=S3)
    sessions, log = 0, []
    while True:
        sessions += 1
        t = [S0 + .05 * (sessions > 1)] * 2
        note = []
        if state["s1"] > 0:
            d = min(state["s1"], H - t[0]); t = [t[0] + d] * 2; state["s1"] -= d
        if state["s1"] <= 1e-9 and state["s2"] > 0:
            d = state["s2"]; t = [t[0] + d] * 2; state["s2"] = 0
        if state["s1"] > 1e-9:
            log.append((sessions, "S1", t[0])); continue
        cpu = max(t)
        s3_end = cpu + state["s3"]; 
        def train(keys, start):
            """Run keys on the GPUs from start; returns the keys done."""
            nonlocal t
            t = start[:]
            done = []
            if dynamic:
                q = list(keys)
                while q:
                    g = 0 if t[0] <= t[1] else 1
                    if t[g] + M * run_h[g] > H:
                        o = 1 - g
                        if t[o] + M * run_h[o] > H: break
                        g = o
                    k = q.pop(0); t[g] += run_h[g]; done.append(k)
            else:
                for g in (0, 1):
                    for k in keys[g::2]:
                        if t[g] + M * run_h[g] > H: break
                        t[g] += run_h[g]; done.append(k)
            return done
        tb = [k for k in train_todo if k[0] == "B"]
        if barrier:
            if tb:
                for k in train(tb, t): train_todo.remove(k)
                note.append(f"B to {max(t):.1f}")
            state["s3"] = max(0, s3_end - max(t)) if s3_end > max(t) else 0
            t = [max(t + [s3_end if state['s3'] >= 0 else 0])] * 2
            state["s3"] = 0 if t[0] <= H else state["s3"]
            if not [k for k in train_todo if k[0] == "B"]:
                rest_todo = [k for k in train_todo if k[0] != "B"]
                if rest_todo:
                    for k in train(rest_todo, t): train_todo.remove(k)
        else:
            # B first, others as soon as a GPU is free and the teachers are done
            q = list(train_todo)
            while q:
                g = 0 if t[0] <= t[1] else 1
                k = next((x for x in q if x[0] == "B" or max(t[g], 0) >= s3_end or state["s3"] == 0), None)
                if k is None:
                    t[g] = s3_end; continue
                if t[g] + M * run_h[g] > H:
                    o = 1 - g
                    if t[o] + M * run_h[o] > H: break
                    g = o
                q.remove(k); train_todo.remove(k); t[g] += run_h[g]
            if s3_end <= H: state["s3"] = 0
        if not train_todo:
            t = [max(t)] * 2 if barrier else t
            q = sorted(score, key=lambda x: -x[1]) if dynamic else score
            while q:
                g = 0 if t[0] <= t[1] else 1
                m, n = q[0]
                if t[g] + M * item_h(g, m, n) > H:
                    o = 1 - g
                    if t[o] + M * item_h(o, m, n) > H: break
                    g = o
                t[g] += item_h(g, m, n); q.pop(0)
            score = q
            if not score:
                log.append((sessions, "done", max(t) + .2)); break
        log.append((sessions, f"runs left {len(train_todo)}, score left {len(score)}", max(t)))
        if sessions > 8: break
    used = sum(x[2] for x in log)
    print(f"{label:<58} sessions {sessions}  GPU-session hours {used:5.1f}  ends: " + " | ".join(f"{x[2]:.1f}" for x in log))
ALL = [f"{a}_{s}" for a in "BCD" for s in range(3)] + ["E_0"]
SEC = ["A", "B_0", "C_0", "D_0"]
asym, sym = (20.3, 16.6), (20.3, 20.3)
ad, ba = (.51, .41), (.337, .27)
ads, bas = (.51, .51), (.337, .337)
for name, st, a_, b_ in (("uneven T4s, as measured", asym, ad, ba), ("both T4s at GPU 0's rate", sym, ads, bas)):
    print("---", name)
    sim(ALL, SEC, st, a_, b_, label="D175 as built (static, B barrier)")
    sim(ALL, SEC, st, a_, b_, dynamic=True, barrier=False, label="D175, scheduler fixed (a free GPU takes the next run)")
    mf = tuple(x * .69 for x in a_)      # reads at the base rate if adapters were merged
    sim(ALL, SEC, st, b_, b_, dynamic=True, barrier=False, label="D175, scheduler fixed + merged adapters")
    two = [r for r in ALL if not r.endswith("_2")]
    sim(two, SEC, st, a_, b_, dynamic=True, barrier=False, label="2 seeds, scheduler fixed")
    sim(two, SEC, st, b_, b_, dynamic=True, barrier=False, label="2 seeds, scheduler fixed + merged adapters")
    sim(ALL, SEC, st, a_, b_, steps=250, dynamic=True, barrier=False, label="4,000 items, scheduler fixed")
    sim(ALL, SEC, st, b_, b_, steps=250, dynamic=True, barrier=False, label="4,000 items, scheduler fixed + merged adapters")
    sim(ALL, SEC, st, a_, b_, steps=250, dynamic=True, barrier=False, ht0=True, v0=True, label="4,000 items, held-out + validation seed 0, sched fixed")
    sim(ALL, SEC, st, a_, b_, dynamic=True, barrier=False, ht0=True, v0=True, label="held-out + validation seed 0, scheduler fixed")
    sim(two, SEC, st, a_, b_, dynamic=True, barrier=False, ht0=True, v0=True, label="2 seeds, held-out + validation seed 0, sched fixed")
print("--- what a reliable two sessions would take")
for name, st, a_, b_ in (("uneven", asym, ad, ba), ("even", sym, ads, bas)):
    sim(two, SEC, st, b_, b_, steps=250, dynamic=True, barrier=False, label=f"{name}: 2 seeds + 4,000 items + scheduler + merged")
    sim(two, SEC, st, a_, b_, steps=250, dynamic=True, barrier=False, label=f"{name}: 2 seeds + 4,000 items + scheduler, not merged")
