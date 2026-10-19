"""Rule-based fault_turn localisation per predicted class.

Rules tuned on train OOF (hit@2, |dt| <= 2):
conflict 0.940 | duplicated_work 0.900 | runaway_loop 0.780 |
deadlock 0.783 | goal_drift 0.752 | dropped_handoff 0.668
"""
from __future__ import annotations

import re
from collections import defaultdict

from engine import events_of
from features4 import wait_parts, sub_eq


def localize(run, label):
    ev = run["events"]
    n = len(ev)
    if not ev or label == "clean":
        return -1
    prod, prod_t = set(), []
    for e in ev:
        if e["kind"] == "deliver" and e["sub"]:
            prod.add(e["sub"])
            prod_t.append((e["t"], e["from"], e["sub"]))
    for a in run["artifacts"]:
        if a.get("subtask") and a.get("status") != "copy":
            prod.add(a["subtask"])
            prod_t.append((a.get("t", 0), a.get("by"), a["subtask"]))
    assigns = [e for e in ev if e["kind"] in ("assign", "assign_maybe") and e["sub"]]

    if label == "dropped_handoff":
        silent, engaged_first = [], []
        for a in assigns:
            s, to, t = a["sub"], a["to"], a["t"]
            if s in prod:
                continue
            onsub = [e for e in ev if e["t"] > t and e["from"] == to and sub_eq(e["sub"], s)]
            kinds = set(e["kind"] for e in onsub) & {
                "work", "deliver", "state_write", "wait", "loop",
                "conflict_msg", "recovery"}
            if kinds:
                continue
            after = [e for e in ev if e["t"] > t and e["from"] == to]
            if not after:
                silent.append(t)
            else:
                engaged_first.append(t)
        if silent:
            return int(silent[0])
        if engaged_first:
            return int(engaged_first[0])
        return int(assigns[0]["t"]) if assigns else (ev[0]["t"])

    if label == "duplicated_work":
        pa = defaultdict(set)
        for t, ag, s in prod_t:
            pa[s].add(ag)
        by_sub = defaultdict(list)
        for a in assigns:
            by_sub[a["sub"]].append(a)
        best = None
        for s, lst in by_sub.items():
            for i in range(1, len(lst)):
                a0, a1 = lst[i - 1], lst[i]
                if a0["from"] != a1["from"]:
                    continue
                if len({a0["to"], a1["to"]} & pa.get(s, set())) >= 2:
                    if best is None or a1["t"] < best:
                        best = a1["t"]
        if best is not None:
            return int(best)
        counts = sorted(by_sub.items(), key=lambda kv: -len(kv[1]))
        if counts and len(counts[0][1]) >= 2:
            return int(counts[0][1][1]["t"])
        return int(ev[len(ev) // 2]["t"])

    if label == "deadlock":
        wp = defaultdict(list)
        for w in events_of(run, "wait"):
            s, tg = wait_parts(w["text"])
            wp[(w["from"], tg or w["to"])].append((w["t"], s))
        pairs = []
        for (a, b), ws in wp.items():
            if a < b and wp.get((b, a)):
                allw = sorted(ws + wp[(b, a)])
                pairs.append((allw[-1][0], allw[0][0], len(allw)))
        if pairs:
            pairs.sort(key=lambda p: (-p[0], p[1]))
            return int(pairs[0][1])
        streak = best_len = best_t = cur = 0
        start = 0
        for e in ev:
            if e["kind"] == "wait":
                if cur == 0:
                    start = e["t"]
                cur += 1
                if cur > best_len:
                    best_len, best_t = cur, start
            else:
                cur = 0
        return int(best_t) if best_len else int(ev[0]["t"])

    if label == "conflict":
        per_key = defaultdict(list)
        for s in run["shared_state"]:
            if s.get("key") == "run_status":
                continue
            per_key[s["key"]].append(s)
        best_t, best_sc = None, -1
        for k, recs in per_key.items():
            ovs = [int(r.get("t", 0)) for r in recs if r.get("op") == "override"]
            vals = [r.get("value") for r in recs]
            alt = sum(1 for i in range(2, len(vals))
                      if vals[i] == vals[i - 2] and vals[i] != vals[i - 1])
            sc = len(ovs) + alt
            if sc > best_sc and ovs:
                best_sc, best_t = sc, min(ovs)
        if best_t is not None:
            return int(best_t)
        cf = events_of(run, "conflict_msg")
        if cf:
            return int(cf[0]["t"])
        return int(ev[0]["t"])

    if label == "goal_drift":
        drifts = events_of(run, "drift")
        if not drifts:
            return int(ev[len(ev) // 2]["t"])
        best_t, best_n = None, -1
        for d in drifts:
            t0 = d["t"]
            early = {e["sub"] for e in ev if e["sub"] and e["t"] <= t0}
            off = sum(1 for e in ev if e["t"] > t0 and e["sub"] and
                      e["sub"] not in early and
                      e["kind"] in ("work", "deliver", "assign"))
            if off > best_n:
                best_n, best_t = off, t0
        return int(best_t if best_t is not None else drifts[0]["t"])

    if label == "runaway_loop":
        loops = events_of(run, "loop")
        if loops:
            import re as _re

            def nrm(t):
                t = (t or "").lower()
                t = _re.sub(r"art_\S+|a\d+", "@", t)
                t = _re.sub(r"\d+", "#", t)
                return _re.sub(r"\s+", " ", t).strip()
            by_key = defaultdict(list)
            for e in loops:
                by_key[(e["from"], e["sub"] or nrm(e["text"]))].append(e)
            best = max(by_key.values(), key=len)
            return int(best[0]["t"])
        return int(ev[len(ev) // 2]["t"])

    return -1


