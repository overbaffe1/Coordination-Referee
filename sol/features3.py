from __future__ import annotations

import numpy as np


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
            if p in (WAIT_PATS[0], WAIT_PATS[1], WAIT_PATS[2], WAIT_PATS[3]):
                return g[1], g[0]          # sub, target
            return g[0], g[1]
    return "", ""


def sub_eq(a, b):
    if not a or not b:
        return False
    if a == b:
        return True
    return a.startswith(b) or b.startswith(a)


def subtask_rows(run):
    """One row per known subtask with its lifecycle features."""
    ev = run["events"]
    n = len(ev)
    roles = {a.get("id"): a.get("role", "") for a in run["agents"]}
    orch = next((aid for aid, r in roles.items() if r == "orchestrator"), None)
    arts = run["artifacts"]

    # production events (deliver msgs + non-copy artifacts)
    prod = []                                  # (t, agent, sub)
    for e in ev:
        if e["kind"] == "deliver" and e["sub"]:
            prod.append((e["t"], e["from"], e["sub"]))
    art_by_sub = defaultdict(list)
    for a in arts:
        if a.get("subtask") and a.get("status") != "copy":
            prod.append((a.get("t", 0), a.get("by"), a["subtask"]))
            art_by_sub[a["subtask"]].append(a)

    assigns = [e for e in ev if e["kind"] in ("assign", "assign_maybe") and e["sub"]]
    subs = set(a["sub"] for a in assigns)
    for t, ag, s in prod:
        if s:
            subs.add(s)

    # wait references (text-based sub + target)
    wait_refs = []                             # (t, waiter, target, sub)
    for e in ev:
        if e["kind"] == "wait":
            s, tg = wait_parts(e["text"])
            wait_refs.append((e["t"], e["from"], tg, s or e["sub"]))

    rows = []
    for s in subs:
        r = {"sub": s}
        a_events = [e for e in assigns if sub_eq(e["sub"], s)]
        prod_e = [(t, ag) for t, ag, ss in prod if sub_eq(ss, s)]
        prod_agents = {ag for _, ag in prod_e}
        r["n_assign"] = len(a_events)
        r["n_assigners"] = len({e["from"] for e in a_events})
        r["n_assignees"] = len({e["to"] for e in a_events})
        r["same_assigner_multi"] = int(len(a_events) >= 2 and
                                       len({e["from"] for e in a_events}) == 1)
        if a_events:
            first = a_events[0]
            r["t_first_assign"] = first["t"]
            r["rel_first_assign"] = first["t"] / max(1, n - 1)
            r["assigner_orch"] = int(first["from"] == orch)
            r["assignee_orch"] = int(first["to"] == orch)
            r["t_last_assign"] = a_events[-1]["t"]
        else:
            r["t_first_assign"] = -1
            r["rel_first_assign"] = -1.0
            r["assigner_orch"] = 0
            r["assignee_orch"] = 0
            r["t_last_assign"] = -1
        # assignee-level delivery
        r["n_producers"] = len(prod_agents)
        r["assignees_that_produced"] = (len({e["to"] for e in a_events} & prod_agents)
                                        if a_events else 0)
        r["produced"] = int(bool(prod_e))
        r["first_assignee_produced"] = int(bool(a_events) and
                                           a_events[0]["to"] in prod_agents)
        r["last_assignee_produced"] = int(bool(a_events) and
                                          a_events[-1]["to"] in prod_agents)
        if len(prod_e) >= 2 and a_events:
            t2 = a_events[-1]["t"]
            d1 = [t for t, ag in prod_e if ag == a_events[0]["to"] and t < t2]
            r["reassign_after_deliv"] = int(bool(d1))
            h0 = None
            for a in arts:
                if sub_eq(a.get("subtask"), s) and a.get("by") == a_events[0]["to"] \
                        and a.get("status") != "copy":
                    h0 = a.get("hash")
            h1 = None
            for a in arts:
                if sub_eq(a.get("subtask"), s) and a.get("by") == a_events[-1]["to"] \
                        and a.get("status") != "copy":
                    h1 = a.get("hash")
            r["dup_hash"] = int(h0 is not None and h1 is not None and h0 == h1)
        else:
            r["reassign_after_deliv"] = 0
            r["dup_hash"] = 0
        r["n_art"] = len(art_by_sub.get(s, []))
        r["art_copy"] = sum(1 for a in art_by_sub.get(s, []) if a.get("status") == "copy")
        # assignee activity on sub after first assign
        if a_events:
            t0, to0 = a_events[0]["t"], a_events[0]["to"]
            on_sub = [e for e in ev if e["t"] > t0 and e["from"] == to0 and sub_eq(e["sub"], s)]
            kc = Counter(e["kind"] for e in on_sub)
            r["assignee_msgs"] = len(on_sub)
            r["assignee_ack"] = kc.get("ack", 0)
            r["assignee_work"] = kc.get("work", 0)
            r["assignee_silent"] = int(len(on_sub) == 0)
            # deliver by assignee after first assign
            r["assignee_deliv_after"] = int(any(
                t > t0 and ag == to0 for t, ag in prod_e))
        else:
            for k in ("assignee_msgs", "assignee_ack", "assignee_work",
                      "assignee_silent", "assignee_deliv_after"):
                r[k] = 0
            r["assignee_silent"] = 0
        # waits referencing this sub
        wr = [(t, w, tg) for t, w, tg, ss in wait_refs if sub_eq(ss, s)]
        r["n_waits_ref"] = len(wr)
        r["waiters"] = len({w for _, w, _ in wr})
        # loops on sub
        lr = [e for e in ev if e["kind"] == "loop" and sub_eq(e["sub"], s)]
        loop_by = Counter(e["from"] for e in lr)
        r["n_loops"] = len(lr)
        r["loop_max_one_sender"] = max(loop_by.values()) if loop_by else 0
        r["loop_senders"] = len(loop_by)
        r["looped_produced"] = int(bool(lr) and bool(prod_e))
        if lr:
            r["loops_after_prod"] = sum(1 for e in lr if e["t"] > max(
                [t for t, _ in prod_e] + [-1]))
        else:
            r["loops_after_prod"] = 0
        r["n_mentions"] = (sum(1 for e in ev if e["sub"] and sub_eq(e["sub"], s))
                           + r["n_art"])
        rows.append(r)
    return rows


def run_level_features(run):
    ev = run["events"]
    n = len(ev)
    f = {}
    kinds = Counter(e["kind"] for e in ev)
    arts = run["artifacts"]
    state = run["shared_state"]

    f["n_msg"] = n
    f["n_agents"] = len(run["agents"])
    f["n_art"] = len(arts)
    for k in ("assign", "deliver", "ack", "work", "wait", "loop", "drift",
              "conflict_msg", "state_write", "recovery", "final", "system"):
        f[f"n_{k}"] = kinds.get(k, 0)
    f["assign_maybe"] = kinds.get("assign_maybe", 0)

    # endings
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

    # conflict
    per_key = defaultdict(list)
    for s in state:
        if s.get("key") == "run_status":
            continue
        per_key[s["key"]].append(s)
    overrides = 0
    hot = 0
    alt = 0
    hot_alt = 0
    for k, recs in per_key.items():
        ovs = [r for r in recs if r.get("op") == "override"]
        overrides += len(ovs)
        hot = max(hot, len(ovs))
        vals = [r.get("value") for r in recs]
        a_ = sum(1 for i in range(2, len(vals))
                 if vals[i] == vals[i - 2] and vals[i] != vals[i - 1])
        alt += a_
        hot_alt = max(hot_alt, a_)
    f["n_overrides"] = overrides
    f["hot_overrides"] = hot
    f["alt_flips"] = alt
    f["hot_alt_flips"] = hot_alt
    f["n_cfmsg"] = kinds.get("conflict_msg", 0)
    f["n_state_writers_max"] = max(
        (len({r.get("agent") for r in recs}) for recs in per_key.values()), default=0)
    f["cfmsg_hot_pair"] = int(f["hot_overrides"] >= 2 and f["n_cfmsg"] >= 2)

    # drift
    drifts = events_of(run, "drift")
    f["n_drift"] = len(drifts)
    if drifts:
        t0 = drifts[0]["t"]
        early = {e["sub"] for e in ev if e["sub"] and e["t"] <= t0}
        late_msgs = [e for e in ev if e["t"] > t0]
        off = [e for e in late_msgs if e["sub"] and e["sub"] not in early]
        f["drift_off_msgs"] = len(off)
        f["drift_off_frac"] = len(off) / max(1, len(late_msgs))
        f["offtopic_art"] = sum(1 for a in arts if a.get("subtask") == "off_topic")
        f["new_subs_after_drift"] = len({e["sub"] for e in off})
        f["drift_in_tail"] = int(t0 >= 0.75 * n)
    else:
        for k in ("drift_off_msgs", "drift_off_frac", "offtopic_art",
                  "new_subs_after_drift", "drift_in_tail"):
            f[k] = 0

    # goal overlap
    import math
    goal_tok = set(re.findall(r"[a-z]{3,}", (run.get("goal") or "").lower()))
    if ev and goal_tok:
        k = max(1, n // 4)
        ov = [len(goal_tok & set(re.findall(r"[a-z]{3,}", e["text"].lower()))) / len(goal_tok)
              for e in ev]
        f["goal_ov_head"] = sum(ov[:k]) / k
        f["goal_ov_tail"] = sum(ov[-k:]) / k
        f["goal_ov_delta"] = f["goal_ov_tail"] - f["goal_ov_head"]
    else:
        f["goal_ov_head"] = f["goal_ov_tail"] = f["goal_ov_delta"] = 0.0

    # loop structure
    loops = events_of(run, "loop")
    f["n_loop"] = len(loops)
    f["loop_in_tail"] = sum(1 for e in loops if e["t"] >= 0.8 * n)
    f["max_identical_text"] = max(
        (Counter((e["from"], re.sub(r"\d+", "#", re.sub(r"art_\S+|a\d+", "@", e["text"].lower())))
                  for e in ev).values()), default=0)

    # wait structure
    waits = events_of(run, "wait")
    f["n_wait"] = len(waits)
    f["wait_tail"] = sum(1 for e in waits if e["t"] >= 0.8 * n)
    wp = defaultdict(list)
    for e in waits:
        s, tg = wait_parts(e["text"])
        wp[(e["from"], tg or e["to"])].append(e)
    mut = 0
    unres_mut = 0
    span_max = 0.0
    for (a, b), ws in wp.items():
        if a < b and (b, a) in wp:
            mut += 1
            allw = ws + wp[(b, a)]
            span = (max(x["t"] for x in allw) - min(x["t"] for x in allw)) / max(1, n - 1)
            span_max = max(span_max, span)
    f["mutual_pairs"] = mut
    f["max_pair_span"] = span_max

    # topology / graph
    tt = run.get("topo_type", "")
    for t in ("star", "pipeline", "mesh", "hierarchical", "blackboard"):
        f[f"topo_{t}"] = float(tt == t)
    f["topo_edges"] = len(run.get("topo_edges", []))
    return f


def extract_all(row):
    run = parse_run(row)
    st_rows = subtask_rows(run)
    rf = run_level_features(run)
    # aggregates over subtasks
    aggf = {}
    def agg(key, fn=np.max):
        vals = [r[key] for r in st_rows] or [0]
        aggf[f"sub_{key}_max"] = float(fn(vals))
    for key in ("n_assign", "n_producers", "assignees_that_produced", "n_waits_ref",
                "n_loops", "loop_max_one_sender", "n_mentions", "assignee_msgs",
                "loops_after_prod", "n_art"):
        vals = [r[key] for r in st_rows] or [0]
        aggf[f"sub_{key}_max"] = float(max(vals))
        aggf[f"sub_{key}_mean"] = float(sum(vals) / len(vals))
    aggf["subs_total"] = len(st_rows)
    aggf["subs_produced"] = sum(r["produced"] for r in st_rows)
    aggf["subs_same_assigner_multi"] = sum(r["same_assigner_multi"] for r in st_rows)
    aggf["subs_reassign_after_deliv"] = sum(r["reassign_after_deliv"] for r in st_rows)
    aggf["subs_dup_hash"] = sum(r["dup_hash"] for r in st_rows)
    aggf["subs_assignee_silent"] = sum(r["assignee_silent"] for r in st_rows)
    aggf["subs_unproduced"] = sum(1 for r in st_rows if not r["produced"])
    aggf["subs_unproduced_real"] = sum(
        1 for r in st_rows if not r["produced"] and r["n_mentions"] >= 2)
    aggf["subs_loop_unprod"] = sum(1 for r in st_rows
                                   if r["n_loops"] >= 3 and not r["produced"])
    aggf["coverage"] = aggf["subs_produced"] / max(1, len(st_rows))
    rf.update(aggf)
    return run, rf, st_rows
