"""Final consolidated feature extraction for Coordination Referee.

Semantic events (engine.py) -> run-level evidence features.
Evidence per fault class + endings + volume + topology. All structural,
robust to vocabulary shift; noise-normalized where possible.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from engine import parse_run, events_of

WAIT_PATS = [
    re.compile(r"waiting for (a\d+) to deliver ([a-z_]+)", re.I),
    re.compile(r"until (a\d+) finishes ([a-z_]+)", re.I),
    re.compile(r"blocked on (a\d+): i need ([a-z_]+) first", re.I),
    re.compile(r"my step depends on (a\d+) closing ([a-z_]+)", re.I),
    re.compile(r"on hold: ([a-z_]+) from (a\d+)", re.I),
    re.compile(r"still holding for (a\d+) on ([a-z_]+)", re.I),
]


def wait_parts(text):
    tl = (text or "").lower()
    for p in WAIT_PATS:
        m = p.search(tl)
        if m:
            g = m.groups()
            if p is WAIT_PATS[4] or p is WAIT_PATS[5]:
                return g[0], g[1]
            return g[1], g[0]
    return "", ""


def sub_eq(a, b):
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a)


def _norm_text(t):
    t = (t or "").lower()
    t = re.sub(r"art_\S+", "@", t)
    t = re.sub(r"\ba\d+\b", "@", t)
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"[^a-z@# ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_features(run):
    ev = run["events"]
    n = len(ev)
    agents = run["agents"]
    arts = run["artifacts"]
    state = run["shared_state"]
    f = {}
    kinds = Counter(e["kind"] for e in ev)
    roles = {a.get("id"): a.get("role", "") for a in agents}
    auth = {aid for aid, r in roles.items() if r in ("orchestrator", "coordinator")}

    # ---------- volume ----------
    f["n_msg"] = n
    f["n_agents"] = len(agents)
    f["n_art"] = len(arts)
    f["n_state"] = len(state)
    for k in ("assign", "assign_maybe", "deliver", "ack", "work", "wait", "loop",
              "drift", "conflict_msg", "state_write", "recovery", "final", "system"):
        f[f"n_{k}"] = kinds.get(k, 0)
        f[f"sh_{k}"] = kinds.get(k, 0) / max(1, n)

    # ---------- subtask structures ----------
    assigns = [e for e in ev if e["kind"] in ("assign", "assign_maybe") and e["sub"]]
    prod = set()
    prod_t = []                      # (t, agent, sub)
    for e in ev:
        if e["kind"] == "deliver" and e["sub"]:
            prod.add(e["sub"])
            prod_t.append((e["t"], e["from"], e["sub"]))
    for a in arts:
        if a.get("subtask") and a.get("status") != "copy":
            prod.add(a["subtask"])
            prod_t.append((a.get("t", 0), a.get("by"), a["subtask"]))
    prod_agents = defaultdict(set)
    for t, ag, s in prod_t:
        prod_agents[s].add(ag)
    assign_by_sub = defaultdict(list)
    for a in assigns:
        assign_by_sub[a["sub"]].append(a)

    # ---- dropped-handoff evidence: unstarted & unproduced subtasks ----
    dh_events = []
    for a in assigns:
        s, to, t = a["sub"], a["to"], a["t"]
        if s in prod:
            continue
        onsub = [e for e in ev if e["t"] > t and e["from"] == to and sub_eq(e["sub"], s)]
        engaged = set(e["kind"] for e in onsub) & {
            "work", "deliver", "state_write", "wait", "loop", "drift",
            "conflict_msg", "recovery"}
        if engaged:
            continue
        dh_events.append(a)
    f["dh_ev"] = len(dh_events)
    f["dh_ev_auth"] = sum(1 for a in dh_events if a["from"] in auth)
    f["dh_ev_rel"] = (dh_events[0]["t"] / max(1, n - 1)) if dh_events else -1.0
    f["dh_silent_all"] = sum(
        1 for a in dh_events
        if not any(e["t"] > a["t"] and e["from"] == a["to"] for e in ev))
    f["subs_unprod"] = len([s for s in assign_by_sub if s not in prod])
    f["subs_total"] = len(assign_by_sub)
    f["coverage"] = 1.0 - f["subs_unprod"] / max(1, f["subs_total"])

    # ---- duplicated-work evidence ----
    art_hash = {}
    for a in arts:
        if a.get("subtask") and a.get("status") != "copy":
            art_hash[(a.get("by"), a["subtask"])] = a.get("hash")
    dw1 = dw2 = dw3 = 0
    for s, lst in assign_by_sub.items():
        for i in range(1, len(lst)):
            a0, a1 = lst[i - 1], lst[i]
            both = {a0["to"], a1["to"]} & prod_agents.get(s, set())
            if a0["from"] != a1["from"]:
                continue
            if len(both) >= 2:
                dw1 += 1
                d1 = [t for t, ag, ss in prod_t if ss == s and ag == a0["to"] and t < a1["t"]]
                if d1:
                    dw2 += 1
                h0, h1 = art_hash.get((a0["to"], s)), art_hash.get((a1["to"], s))
                if h0 and h0 == h1:
                    dw3 += 1
    f["dw_ev"] = dw1
    f["dw_after_deliv"] = dw2
    f["dw_same_hash"] = dw3
    f["n_reassign_same"] = sum(
        1 for s, lst in assign_by_sub.items() for i in range(1, len(lst))
        if lst[i]["from"] == lst[i - 1]["from"])
    f["max_assign_per_sub"] = max((len(v) for v in assign_by_sub.values()), default=0)

    # ---- waits / deadlock ----
    waits = events_of(run, "wait")
    wparts = [(w["t"], w["from"], *wait_parts(w["text"])) for w in waits]
    wpair = defaultdict(list)
    for t, frm, s, tg in wparts:
        wpair[(frm, tg or "")].append((t, s))
    mut_pairs = []
    for (a, b), ws in wpair.items():
        if a < b and wpair.get((b, a)):
            allw = sorted(ws + wpair[(b, a)])
            unf = 0
            for tt, frm, s, tg in [(x[0], k[0], x[1], k[1]) for k in ((a, b), (b, a)) for x in wpair[k]]:
                if s and not any(ag == tg for _, ag, ss in prod_t if ss == s):
                    unf += 1
            mut_pairs.append({
                "first_t": allw[0][0], "last_t": allw[-1][0], "n": len(allw),
                "unres": min(unf, 2)})
    f["n_mut_pairs"] = len(mut_pairs)
    f["mut_unres"] = sum(1 for p in mut_pairs if p["unres"] >= 1)
    f["mut_unres2"] = sum(1 for p in mut_pairs if p["unres"] >= 2)
    f["mut_max_n"] = max((p["n"] for p in mut_pairs), default=0)
    f["mut_max_span"] = max(((p["last_t"] - p["first_t"]) / max(1, n - 1)
                             for p in mut_pairs), default=0.0)
    f["mut_max_last_frac"] = max((p["last_t"] / max(1, n - 1) for p in mut_pairs),
                                 default=0.0)
    f["n_wait"] = len(waits)
    tail0 = int(n * 0.8)
    f["wait_tail"] = sum(1 for w in waits if w["t"] >= tail0)
    last_prod = max([t for t, _, _ in prod_t] + [-1])
    f["wait_after_prod"] = sum(1 for w in waits if w["t"] > last_prod)

    # ---- conflict ----
    per_key = defaultdict(list)
    for s in state:
        if s.get("key") == "run_status":
            continue
        per_key[s["key"]].append(s)
    hot_ov = 0
    hot_alt = 0
    n_ov = 0
    hot_t = -1
    for k, recs in per_key.items():
        ovs = [r for r in recs if r.get("op") == "override"]
        n_ov += len(ovs)
        vals = [r.get("value") for r in recs]
        alt = sum(1 for i in range(2, len(vals))
                  if vals[i] == vals[i - 2] and vals[i] != vals[i - 1])
        if (len(ovs) + alt) > (hot_ov + hot_alt):
            hot_ov, hot_alt = len(ovs), alt
            hot_t = min([int(r.get("t", 0)) for r in ovs] + [10 ** 9])
    f["hot_overrides"] = hot_ov
    f["hot_alt"] = hot_alt
    f["n_overrides"] = n_ov
    f["cf_hot_pair"] = int(hot_ov >= 2 and kinds.get("conflict_msg", 0) >= 1)
    f["cf_strong"] = int(hot_ov >= 2 and hot_alt >= 1 and kinds.get("conflict_msg", 0) >= 2)
    f["ov_tail"] = sum(1 for s in state if s.get("op") == "override"
                       and s.get("t", 0) >= int(len(state) * 0.8))
    f["cfmsg"] = kinds.get("conflict_msg", 0)

    # ---- drift ----
    drifts = events_of(run, "drift")
    f["n_drift"] = len(drifts)
    goal_tok = set(re.findall(r"[a-z]{3,}", (run.get("goal") or "").lower()))
    if drifts:
        t0 = drifts[0]["t"]
        early = {e["sub"] for e in ev if e["sub"] and e["t"] <= t0}
        late = [e for e in ev if e["t"] > t0]
        off = [e for e in late if e["sub"] and e["sub"] not in early
               and e["kind"] in ("work", "deliver", "assign", "ack", "loop")]
        f["gd_off_work"] = len(off)
        f["gd_off_frac"] = len(off) / max(1, len(late))
        f["gd_new_subs"] = len({e["sub"] for e in off})
        f["gd_drift_after"] = sum(1 for d in drifts if d["t"] > t0)
        f["gd_first_rel"] = t0 / max(1, n - 1)
    else:
        for k in ("gd_off_work", "gd_off_frac", "gd_new_subs", "gd_drift_after",
                  "gd_first_rel"):
            f[k] = 0
    f["offtopic_art"] = sum(1 for a in arts if a.get("subtask") == "off_topic")
    if ev and goal_tok:
        k = max(1, n // 4)
        ov = [len(goal_tok & set(re.findall(r"[a-z]{3,}", e["text"].lower()))) /
              len(goal_tok) for e in ev]
        f["goal_ov_head"] = sum(ov[:k]) / k
        f["goal_ov_tail"] = sum(ov[-k:]) / k
        f["goal_ov_delta"] = f["goal_ov_tail"] - f["goal_ov_head"]
        f["goal_ov_zero_tail"] = sum(1 for x in ov[-k:] if x == 0) / k
    else:
        f["goal_ov_head"] = f["goal_ov_tail"] = f["goal_ov_delta"] = 0.0
        f["goal_ov_zero_tail"] = 0.0

    # ---- loop ----
    loops = events_of(run, "loop")
    by_key = defaultdict(list)
    for e in loops:
        by_key[(e["from"], e["sub"] or _norm_text(e["text"]))].append(e)
    f["n_loop"] = len(loops)
    f["loop_max_series"] = max((len(v) for v in by_key.values()), default=0)
    f["loop_series3"] = sum(1 for v in by_key.values() if len(v) >= 3)
    f["loop_tail"] = sum(1 for e in loops if e["t"] >= tail0)
    f["loop_unprod_series"] = sum(
        1 for (fr, s), v in by_key.items()
        if len(v) >= 3 and not (prod_agents.get(s, set()) & {fr}))
    cnt = Counter((e["from"], _norm_text(e["text"])) for e in ev)
    f["max_identical"] = max(cnt.values()) if cnt else 0
    best = cur = 0
    prev = None
    for e in ev:
        x = _norm_text(e["text"])
        cur = cur + 1 if x == prev else 1
        best = max(best, cur)
        prev = x
    f["max_consec_identical"] = best

    # ---------- endings ----------
    sys_texts = [e["text"].lower() for e in ev if e["kind"] == "system"]
    f["sys_no_progress"] = int(any("no progress" in t for t in sys_texts))
    f["sys_budget"] = int(any("budget" in t or "terminated" in t for t in sys_texts))
    f["sys_halt"] = int(any(("halting" in t or "stopping" in t) and
                            "no progress" not in t and "budget" not in t
                            for t in sys_texts))
    st = [s for s in state if s.get("key") == "run_status"]
    v = st[-1].get("value") if st else "none"
    for val in ("completed", "timeout", "aborted", "incomplete", "none"):
        f[f"rs_{val}"] = int(v == val)
    fd = [a for a in arts if a.get("subtask") == "final_delivery"]
    f["fd_present"] = int(bool(fd))
    f["fd_final"] = int(bool(fd) and fd[0].get("status") == "final")
    f["fd_partial"] = int(bool(fd) and fd[0].get("status") == "partial")
    f["has_final_msg"] = int(kinds.get("final", 0) > 0)
    f["prod_end_frac"] = last_prod / max(1, n - 1)
    f["quiet_tail"] = (n - 1 - last_prod) / max(1, n)
    hashes = Counter(a.get("hash") for a in arts if a.get("status") != "copy")
    f["dup_hash_pairs"] = sum(v - 1 for v in hashes.values() if v > 1)
    f["art_copy"] = sum(1 for a in arts if a.get("status") == "copy")
    f["art_partial"] = sum(1 for a in arts if a.get("status") == "partial")

    # ---------- topology / graph ----------
    tt = run.get("topo_type", "")
    for t in ("star", "pipeline", "mesh", "hierarchical", "blackboard"):
        f[f"topo_{t}"] = float(tt == t)
    allowed = {frozenset(e) for e in run.get("topo_edges", []) if len(e) == 2}
    pairs = set()
    viol = 0
    for e in ev:
        if e["from"] and e["to"] and e["from"] != e["to"]:
            pr = frozenset((e["from"], e["to"]))
            pairs.add(pr)
            if allowed and pr not in allowed:
                viol += 1
    f["edge_viol_share"] = viol / max(1, len(pairs))
    f["topo_edges"] = len(run.get("topo_edges", []))

    # sequence n-grams of kinds (cheap temporal signal)
    seq = [e["kind"] for e in ev]
    for g in (2, 3):
        c = Counter(tuple(seq[i:i + g]) for i in range(len(seq) - g + 1))
        f[f"ngram{g}_top1"] = c.most_common(1)[0][1] if c else 0
    # sender concentration in tail
    tail = ev[tail0:]
    tc = Counter(e["from"] for e in tail)
    f["tail_sender_max"] = max(tc.values()) / max(1, len(tail)) if tail else 0
    return f


def featurize_row(row):
    run = parse_run(row)
    return run, extract_features(run)
