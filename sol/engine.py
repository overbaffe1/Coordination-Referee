"""Semantic engine for Coordination Referee.

Parses one run into typed events using the simulator's message templates.
Designed to be robust to the dataset noise:
  * truncated texts (12-30 chars) -> prefix matching;
  * stripped refs (``art_x`` / missing) -> template decides assign vs deliver;
  * empty intents -> subtask recovered from the message text.

Message semantics (from docs/ontology.md + template mining):
  handoff  : assign (no refs / assign template) | deliver (refs / deliver template)
  inform   : work | ack | ack_small | loop | drift
  status   : wait (always carries waiter -> target + subtask)
  state_update : state_write | conflict_msg
  final    : run summary   | system : run stop
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

AG = r"(a\d+)"
SUB = r"([a-z_]+)"
ART = r"(art_\w*)"
NUM = r"(\d+)"

# ---------------------------------------------------------------- templates
# handoff: delivery phrasing (checked BEFORE assign; refs may be stripped)
DELIVER_PATTERNS = [
    re.compile(r"^output for " + SUB + r" ready", re.I),
    re.compile(r"^delivering " + ART + r" for " + SUB, re.I),
    re.compile(r"^finished " + SUB + r"\. result", re.I),
    re.compile(r"^" + SUB + r" is done, artifact", re.I),
    re.compile(r"^" + SUB + r" complete, see", re.I),
    re.compile(r"^" + SUB + r" closed out;", re.I),
    re.compile(r"^done: " + SUB, re.I),
    re.compile(r"^result ready for " + SUB, re.I),
]
# handoff: recovery / reassignment phrasing
RECOVERY_PATTERNS = [
    re.compile(r"^recovery", re.I),
    re.compile(r"^escalating", re.I),
    re.compile(r"^mitigation", re.I),
    re.compile(r"^repairing", re.I),
    re.compile(r"^fallback", re.I),
    re.compile(r"^detected stall", re.I),
    re.compile(r"^re-?assign", re.I),
    re.compile(r"^retry", re.I),
    re.compile(r"^takes over", re.I),
]
# handoff: assignment phrasing
ASSIGN_PATTERNS = [
    re.compile(r"^assigning " + SUB + r" to", re.I),
    re.compile(r"^" + AG + r": your subtask is " + SUB, re.I),
    re.compile(r"^task allocation: " + SUB + r" ->", re.I),
    re.compile(r"^handing " + SUB + r" over to", re.I),
    re.compile(r"^" + AG + r" owns " + SUB + r" from this point", re.I),
    re.compile(r"^" + AG + r", please take care of " + SUB, re.I),
]
# inform: loop phrasing
LOOP_PATTERNS = [
    re.compile(r"^re-?checking ", re.I),
    re.compile(r"^another pass over ", re.I),
    re.compile(r"^iterating on ", re.I),
    re.compile(r"^looping back on ", re.I),
    re.compile(r"^one more attempt at ", re.I),
    re.compile(r"^repeating ", re.I),
    re.compile(r"^still looping on ", re.I),
    re.compile(r"^another iteration of ", re.I),
]
# inform: drift phrasing
DRIFT_PATTERNS = [
    re.compile(r"^side note:", re.I),
    re.compile(r"^while here, let us also", re.I),
    re.compile(r"^expanding scope", re.I),
    re.compile(r"^deprioriti[sz]ing", re.I),
    re.compile(r"^new direction:", re.I),
    re.compile(r"^pivoting (toward|to)", re.I),
    re.compile(r"^switching focus", re.I),
    re.compile(r"^also look into", re.I),
    re.compile(r"^matters more", re.I),
]
# inform: ack-of-task phrasing (assignee accepted the work)
ACK_PATTERNS = [
    re.compile(r"^acknowledged, starting ", re.I),
    re.compile(r"^taking ", re.I),
    re.compile(r"^got it, working on ", re.I),
    re.compile(r"^confirmed\. ", re.I),
    re.compile(r"^roger, ", re.I),
    re.compile(r"^" + SUB + r" accepted, eta", re.I),
]
# status: wait phrasing; capture target agent + subtask
WAIT_PATTERNS = [
    re.compile(r"^waiting for " + AG + r" to deliver " + SUB, re.I),
    re.compile(r"^cannot proceed (with my part )?until " + AG + r" finishes " + SUB, re.I),
    re.compile(r"^blocked on " + AG + r": i need " + SUB + r" first", re.I),
    re.compile(r"^my step depends on " + AG + r" closing " + SUB, re.I),
    re.compile(r"^on hold: " + SUB + r" from " + AG + r" has not arrived", re.I),
    re.compile(r"^still holding for " + AG + r" on " + SUB, re.I),
]
# state_update: conflict phrasing
CONFLICT_PATTERNS = [
    re.compile(r"^that contradicts my finding", re.I),
    re.compile(r"^overriding ", re.I),
    re.compile(r"^i disagree:", re.I),
    re.compile(r"^conflict detected", re.I),
    re.compile(r"^rejecting the current ", re.I),
    re.compile(r"=" + NUM + r" is incorrect, switching to " + NUM, re.I),
]
# state_update: plain write phrasing
WRITE_PATTERNS = [
    re.compile(r"^recorded ", re.I),
    re.compile(r"^shared state updated", re.I),
    re.compile(r"^setting ", re.I),
    re.compile(r"^writing ", re.I),
    re.compile(r"^state patch applied", re.I),
]

STOP_SUBTASKS = {"status", "finalize", ""}


def _match(patterns, text: str):
    for p in patterns:
        m = p.search(text)
        if m:
            return m
    return None


def msg_kind(m: Dict[str, Any]) -> str:
    """Classify one message into a semantic kind."""
    t, text = m.get("type", ""), (m.get("text") or "")
    tl = text.lower()
    if t == "handoff":
        if _match(DELIVER_PATTERNS, tl) or m.get("refs"):
            return "deliver"
        if _match(RECOVERY_PATTERNS, tl):
            return "recovery"
        if _match(ASSIGN_PATTERNS, tl):
            return "assign"
        # fallback for heavily truncated assign texts
        if not m.get("refs"):
            return "assign_maybe"
        return "deliver_maybe"
    if t == "status":
        return "wait"
    if t == "state_update":
        if _match(CONFLICT_PATTERNS, tl):
            return "conflict_msg"
        return "state_write"
    if t == "inform":
        if _match(DRIFT_PATTERNS, tl):
            return "drift"
        if _match(LOOP_PATTERNS, tl):
            return "loop"
        if _match(ACK_PATTERNS, tl):
            return "ack"
        return "work"          # work progress / small acks / anything else
    if t == "final":
        return "final"
    if t == "system":
        return "system"
    return "other"


def subtask_of(m: Dict[str, Any], kind: str) -> str:
    """Recover the subtask name of a message (intent first, then text)."""
    it = (m.get("intent") or "").strip()
    if it and it not in STOP_SUBTASKS:
        return it
    text = (m.get("text") or "").lower()
    kind_pats = {
        "deliver": [r"^output for (" + SUB + r") ready", r"^delivering " + ART + r" for (" + SUB + r")",
                    r"^finished (" + SUB + r")\. result", r"^(" + SUB + r") is done",
                    r"^(" + SUB + r") complete, see", r"^(" + SUB + r") closed out",
                    r"^done: (" + SUB + r")", r"^result ready for (" + SUB + r")"],
        "assign": [r"^assigning (" + SUB + r") to", r": your subtask is (" + SUB + r")",
                   r"^task allocation: (" + SUB + r") ->", r"^handing (" + SUB + r") over",
                   r"owns (" + SUB + r") from", r"please take care of (" + SUB + r")",
                   r"^recovery: re-?assign(?:ing)? (" + SUB + r")"],
        "wait": [r"to deliver (" + SUB + r")", r"finishes (" + SUB + r")",
                 r"i need (" + SUB + r") first", r"closing (" + SUB + r")",
                 r"^on hold: (" + SUB + r") from", r"^still holding for .* on (" + SUB + r")"],
        "loop": [r"^re-?checking (" + SUB + r")", r"^another pass over (" + SUB + r")",
                 r"^iterating on (" + SUB + r")", r"^looping back on (" + SUB + r")",
                 r"^one more attempt at (" + SUB + r")", r"^repeating (" + SUB + r")",
                 r"^still looping on (" + SUB + r")", r"^another iteration of (" + SUB + r")"],
        "ack": [r"^acknowledged, starting (" + SUB + r")", r"^taking (" + SUB + r")",
                r"^got it, working on (" + SUB + r")", r"^confirmed\. (" + SUB + r")",
                r"^roger, (" + SUB + r")", r"^(" + SUB + r") accepted, eta"],
        "recovery": [r"re-?assign(?:ing)? (" + SUB + r")", r"^escalating (" + SUB + r")",
                     r"^mitigation(?: plan)? (?:for )?(" + SUB + r")",
                     r"^repairing (" + SUB + r")", r"^fallback (?:for )?(" + SUB + r")"],
    }
    for p in kind_pats.get(kind, []):
        mm = re.search(p, text)
        if mm and mm.group(1) and mm.group(1) not in STOP_SUBTASKS:
            return mm.group(1)
    return ""


def parse_run(row: Dict[str, Any]) -> Dict[str, Any]:
    """Decode JSON columns and pre-compute typed events for one run."""
    run = {"run_id": row.get("run_id"), "goal": row.get("goal", "") or ""}
    for f in ("agents", "shared_state", "messages", "artifacts", "topology"):
        v = row.get(f)
        run[f] = json.loads(v) if isinstance(v, str) else (v if v is not None else [])
    if isinstance(run["topology"], dict):
        run["topo_type"] = run["topology"].get("type", "")
        run["topo_edges"] = run["topology"].get("edges", []) or []
    else:
        run["topo_type"], run["topo_edges"] = "", []
    run["agent_ids"] = [a.get("id") for a in run["agents"]]

    msgs = run["messages"]
    events = []
    for i, m in enumerate(msgs):
        k = msg_kind(m)
        events.append({
            "i": i, "t": int(m.get("t", i)), "from": m.get("from"), "to": m.get("to"),
            "kind": k, "sub": subtask_of(m, k), "refs": m.get("refs") or [],
            "text": m.get("text") or "", "tool": m.get("tool") or "",
            "raw": m,
        })
    run["events"] = events
    return run


def events_of(run, kind):
    return [e for e in run["events"] if e["kind"] == kind]
