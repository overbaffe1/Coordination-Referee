"""Candidate fault detectors built on top of the semantic engine.

Each detector returns a dict:
    {cls, strength, turns}   turns = candidate fault_turns (ordered by evidence)
No arbitration here - the arbiter (built from train statistics) decides.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from engine import events_of


def _norm_text(t: str) -> str:
    import re
    t = t.lower()
    t = re.sub(r"art_\S+", "@", t)
    t = re.sub(r"\ba\d+\b", "@", t)
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"\bwith @\b", "with @ ", t)
    t = re.sub(r"[^a-z@# ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def delivered_map(run) -> Dict[str, set]:
    """agent -> set of subtasks it delivered (deliver msgs + artifacts)."""
    d = defaultdict(set)
    for e in run["events"]:
        if e["kind"] == "deliver" and e["sub"]:
            d[e["from"]].add(e["sub"])
    for a in run["artifacts"]:
        if a.get("subtask"):
            d[a.get("by")].add(a["subtask"])
    return d


def artifact_subs_by(run) -> Dict[str, set]:
    d = defaultdict(set)
    for a in run["artifacts"]:
        if a.get("subtask"):
            d[a.get("by")].add(a["subtask"])
    return d


# ------------------------------------------------------------------ dropped
def detect_dropped(run) -> Dict[str, Any]:
    assigns = [e for e in run["events"] if e["kind"] in ("assign", "assign_maybe")]
    deliv = delivered_map(run)
    undelivered = []
    for a in assigns:
        if a["sub"] and a["sub"] not in deliv.get(a["to"], set()):
            undelivered.append(a)
    n_und = len(undelivered)
    turns = sorted(set(a["t"] for a in undelivered))
    return {"cls": "dropped_handoff", "strength": float(n_und),
            "n_assign": len(assigns), "turns": turns}


# --------------------------------------------------------------- duplicated
def detect_duplicated(run) -> Dict[str, Any]:
    assigns = [e for e in run["events"] if e["kind"] == "assign"]
    by_sub = defaultdict(list)
    for a in assigns:
        if a["sub"]:
            by_sub[a["sub"]].append(a)
    deliv = delivered_map(run)
    dups, turns = [], []
    for sub, lst in by_sub.items():
        if len(lst) >= 2:
            owners = {a["to"] for a in lst}
            # each owner (or at least two owners) produced an artifact?
            prod = sum(1 for o in owners if sub in deliv.get(o, set()))
            if prod >= 2:
                dups.append(sub)
                turns.append(lst[1]["t"])
    return {"cls": "duplicated_work", "strength": float(len(dups)),
            "n_multi_assign": sum(1 for v in by_sub.values() if len(v) > 1),
            "turns": sorted(turns)}


# ----------------------------------------------------------------- deadlock
def detect_deadlock(run) -> Dict[str, Any]:
    waits = events_of(run, "wait")
    delivers = events_of(run, "deliver")
    art = run["artifacts"]
    last_deliv_t = max([e["t"] for e in delivers] + [a.get("t", 0) for a in art] + [-1])
    # mutual pairs (A waits B) x (B waits A)
    pair_waits = defaultdict(list)   # (a,b) -> wait events a->b
    for w in waits:
        pair_waits[(w["from"], w["to"])].append(w)
    pairs = []
    for (a, b), ws in pair_waits.items():
        if a < b:
            rev = pair_waits.get((b, a), [])
            if rev:
                allw = ws + rev
                pairs.append((a, b, allw))
    cands = []
    for a, b, allw in pairs:
        first_t = min(w["t"] for w in allw)
        # resolution: any deliver after first_t?
        resolved = any(d["t"] > first_t for d in delivers)
        tail_art = last_deliv_t > first_t
        cands.append({"a": a, "b": b, "first_t": first_t, "n_waits": len(allw),
                      "resolved": resolved, "art_after": tail_art,
                      "score": 0 if resolved else 1})
    # strength: number of unresolved mutual pairs + total wait mass
    unres = [c for c in cands if c["score"] > 0]
    turns = sorted(set(c["first_t"] for c in unres))
    return {"cls": "deadlock", "strength": float(len(unres)),
            "n_pairs": len(cands), "n_waits": len(waits),
            "last_deliv_t": last_deliv_t, "turns": turns}


# ----------------------------------------------------------------- conflict
def detect_conflict(run) -> Dict[str, Any]:
    per_key = defaultdict(list)
    for s in run["shared_state"]:
        if s.get("key") == "run_status":
            continue
        per_key[s["key"]].append(s)
    hot, hot_n, hot_first_ov = None, 0, None
    n_over = 0
    for k, recs in per_key.items():
        ovs = [r for r in recs if r.get("op") == "override"]
        n_over += len(ovs)
        if len(ovs) >= 2 and len(ovs) > hot_n:
            hot, hot_n = k, len(ovs)
            hot_first_ov = min(int(r.get("t", 0)) for r in ovs)
    n_conf_msgs = len(events_of(run, "conflict_msg"))
    flips = 0
    for k, recs in per_key.items():
        vals = [r.get("value") for r in recs]
        flips += sum(1 for i in range(1, len(vals)) if vals[i] != vals[i-1])
    strength = 0.0
    turns = []
    if hot is not None:
        strength = 1.0 + hot_n
        turns = [hot_first_ov]
    return {"cls": "conflict", "strength": strength, "hot_key": hot,
            "n_overrides": n_over, "n_conf_msgs": n_conf_msgs, "flips": flips,
            "turns": turns}


# --------------------------------------------------------------- goal drift
def detect_drift(run) -> Dict[str, Any]:
    drifts = events_of(run, "drift")
    goal = (run.get("goal") or "").lower()
    import re
    goal_tok = set(re.findall(r"[a-z]{3,}", goal))
    arts = run["artifacts"]
    off_topic_art = [a for a in arts if a.get("subtask") == "off_topic"]
    # new intents appearing after first drift message
    subs_seq = [(e["t"], e["sub"]) for e in run["events"] if e["sub"]]
    early_subs, late_subs = set(), set()
    if subs_seq:
        first_drift_t = drifts[0]["t"] if drifts else 10**9
        for t, s in subs_seq:
            (late_subs if t >= first_drift_t else early_subs).add(s)
    new_after = late_subs - early_subs
    strength = float(len(drifts)) + 2.0 * len(off_topic_art) + 1.5 * len(new_after)
    turns = [d["t"] for d in drifts]
    return {"cls": "goal_drift", "strength": strength, "n_drift": len(drifts),
            "n_off_topic_art": len(off_topic_art), "new_after": len(new_after),
            "turns": turns}


# ------------------------------------------------------------- runaway loop
def detect_loop(run) -> Dict[str, Any]:
    loops = events_of(run, "loop")
    by_key = defaultdict(list)
    for e in loops:
        by_key[(e["from"], e["sub"] or _norm_text(e["text"]))].append(e)
    max_rep = max((len(v) for v in by_key.values()), default=0)
    n_rep_keys = sum(1 for v in by_key.values() if len(v) >= 3)
    turns = []
    for k, v in by_key.items():
        if len(v) >= 3:
            turns.append(v[0]["t"])
    turns = sorted(turns)
    return {"cls": "runaway_loop", "strength": float(max_rep), "max_rep": max_rep,
            "n_rep_keys": n_rep_keys, "n_loops": len(loops), "turns": turns}


DETECTORS = [detect_dropped, detect_duplicated, detect_deadlock,
             detect_conflict, detect_drift, detect_loop]


def all_detections(run) -> Dict[str, Dict[str, Any]]:
    return {d(run)["cls"]: d(run) for d in DETECTORS}
