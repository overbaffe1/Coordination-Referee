"""Anchor-level candidate generation + forward-persistence features.

The simulator injects the fault at fault_turn and the fault's behaviour
persists until the end of the run. For every plausible anchor event we
describe what happens AFTER it. Per-class binary models score anchors;
run-level aggregation produces the final prediction.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from engine import parse_run, events_of

CLASSES = ["dropped_handoff", "duplicated_work", "deadlock",
           "conflict", "goal_drift", "runaway_loop"]


class RunContext:
    """Precomputed structures for one run."""

    def __init__(self, run):
        self.run = run
        ev = run["events"]
        self.ev = ev
        self.n = len(ev)
        self.assigns = [e for e in ev if e["kind"] in ("assign", "assign_maybe") and e["sub"]]
        self.delivers = [e for e in ev if e["kind"] == "deliver" and e["sub"]]
        self.waits = events_of(run, "wait")
        self.loops = events_of(run, "loop")
        self.drifts = events_of(run, "drift")
        self.cfmsgs = events_of(run, "conflict_msg")
        # production times: (t, agent, sub)
        self.prod = []
        for e in self.delivers:
            self.prod.append((e["t"], e["from"], e["sub"]))
        for a in run["artifacts"]:
            if a.get("subtask") and a.get("status") != "copy":
                self.prod.append((a.get("t", 0), a.get("by"), a["subtask"]))
        self.prod_by_sub = defaultdict(list)
        for t, ag, s in self.prod:
            self.prod_by_sub[s].append((t, ag))
        self.prod_agents = defaultdict(set)
        for t, ag, s in self.prod:
            self.prod_agents[s].add(ag)
        self.last_prod_t = max([t for t, _, _ in self.prod] + [-1])
        # assigns per sub
        self.assign_by_sub = defaultdict(list)
        for a in self.assigns:
            self.assign_by_sub[a["sub"]].append(a)
        # state records per key (with ops)
        self.state = [s for s in run["shared_state"] if s.get("key") != "run_status"]
        self.state_by_key = defaultdict(list)
        for s in self.state:
            self.state_by_key[s["key"]].append(s)
        # mutual wait partner info
        self.wait_pair = defaultdict(list)   # (from,to) -> waits
        for w in self.waits:
            self.wait_pair[(w["from"], w["to"])].append(w)
        self.hash_by = {}
        for a in run["artifacts"]:
            if a.get("subtask") and a.get("status") != "copy":
                self.hash_by[(a.get("by"), a["subtask"])] = a.get("hash")

    # ---------- helpers ----------
    def produced_after(self, t, sub=None, agent=None):
        out = []
        for tt, ag, s in self.prod:
            if tt > t and (sub is None or s == sub) and (agent is None or ag == agent):
                out.append((tt, ag, s))
        return out

    def prod_times(self, sub, agent):
        return sorted(t for t, ag, s in self.prod if s == sub and ag == agent)


def anchor_candidates(ctx) -> list:
    """Yield dicts {cls, t, base features} for every plausible anchor."""
    ev, n = ctx.ev, ctx.n
    out = []

    def after(t):
        return [e for e in ev if e["t"] > t]

    def common(e, cls):
        t = e["t"]
        aft = after(t)
        return {"cls": cls, "t": t, "rel_pos": t / max(1, n - 1),
                "n_after": len(aft)}

    # ---- dropped_handoff: assigns whose sub never produced by assignee
    for a in ctx.assigns:
        t = a["t"]
        s, to = a["sub"], a["to"]
        f = common(a, "dropped_handoff")
        f["assignee_prod_later"] = int(bool(ctx.prod_times(s, to)))
        f["any_prod_later"] = int(bool(ctx.produced_after(t, s)))
        f["any_prod_later_any_sub"] = int(bool(ctx.produced_after(t)))
        after_e = [e for e in ev if e["t"] > t and e["from"] == to]
        f["assignee_msgs_after"] = len(after_e)
        kinds = defaultdict(int)
        for e in after_e:
            kinds[e["kind"]] += 1
        f["assignee_work_after"] = kinds.get("work", 0)
        f["assignee_ack_after"] = kinds.get("ack", 0)
        f["assignee_deliv_other_after"] = kinds.get("deliver", 0)
        f["assignee_wait_after"] = kinds.get("wait", 0)
        f["sub_wait_after"] = sum(1 for w in ctx.waits if w["t"] > t and w["sub"] == s)
        f["sub_loop_after"] = sum(1 for l in ctx.loops if l["t"] > t and l["sub"] == s)
        f["n_assigns_sub"] = len(ctx.assign_by_sub.get(s, []))
        f["to_silent_rest"] = int(len(after_e) == 0)
        out.append(f)

    # ---- duplicated_work: re-assigns (same-from) of produced subtasks
    for s, lst in ctx.assign_by_sub.items():
        for i in range(1, len(lst)):
            a0, a1 = lst[i - 1], lst[i]
            f = common(a1, "duplicated_work")
            t2 = a1["t"]
            f["same_from"] = int(a0["from"] == a1["from"])
            f["same_to"] = int(a0["to"] == a1["to"])
            prods = {a0["to"], a1["to"]} & ctx.prod_agents.get(s, set())
            f["both_produced"] = int(len(prods) >= 2)
            d1 = ctx.prod_times(s, a0["to"])
            f["first_prod_before"] = int(bool([x for x in d1 if x < t2]))
            f["gap_from_first_assign"] = t2 - a0["t"]
            d2 = ctx.prod_times(s, a1["to"])
            f["second_prod_after"] = int(bool([x for x in d2 if x > t2]))
            h0 = ctx.hash_by.get((a0["to"], s))
            h1 = ctx.hash_by.get((a1["to"], s))
            f["same_hash"] = int(h0 is not None and h0 == h1 and h0 is not None)
            f["n_prods_sub"] = len(ctx.prod_agents.get(s, set()))
            f["n_assigns_sub"] = len(lst)
            f["recovery_between"] = int(any(
                e["kind"] == "recovery" and a0["t"] < e["t"] < t2 for e in ev))
            out.append(f)

    # ---- deadlock: waits that have a mutual partner
    for w in ctx.waits:
        rev = ctx.wait_pair.get((w["to"], w["from"]), [])
        if not rev:
            continue
        f = common(w, "deadlock")
        t = w["t"]
        f["mutual_n"] = len(rev)
        f["partner_wait_after"] = int(any(x["t"] > t for x in rev))
        prods_after = ctx.produced_after(t)
        f["any_prod_after"] = int(bool(prods_after))
        f["n_prod_after"] = len(prods_after)
        f["sub_prod_later_by_target"] = int(bool(
            [x for x in ctx.prod_times(w["sub"], w["to"]) if x > t]))
        f["pair_all_t"] = len(ctx.wait_pair[(w["from"], w["to"])]) + len(rev)
        waits_after = [x for x in ctx.waits if x["t"] > t]
        f["n_waits_after"] = len(waits_after)
        f["waits_after_frac_of_run"] = len(waits_after) / max(1, ctx.n - t)
        f["pair_waits_after"] = sum(1 for x in ctx.wait_pair[(w["from"], w["to"])] + rev
                                    if x["t"] > t)
        f["pair_is_unresolved"] = int(not any(
            x for x in ctx.prod_times(w["sub"], w["to"]) if x > t) and
            not any(x for x in ctx.prod_times(
                # the reverse direction's awaited sub
                next((r["sub"] for r in rev), ""), w["from"]) if x > t))
        f["last_prod_after"] = int(ctx.last_prod_t > t)
        out.append(f)

    # ---- conflict: conflict msgs and override records
    conflict_anchors = [(e["t"], e["sub"], "msg") for e in ctx.cfmsgs]
    for s in ctx.state:
        if s.get("op") == "override":
            conflict_anchors.append((s.get("t", 0), s.get("key"), "rec"))
    seen = set()
    for t, key, src in conflict_anchors:
        if (t, key) in seen:
            continue
        seen.add((t, key))
        f = {"cls": "conflict", "t": t, "rel_pos": t / max(1, ctx.n - 1),
             "n_after": ctx.n - t - 1}
        recs = ctx.state_by_key.get(key, [])
        ovs = [r for r in recs if r.get("op") == "override"]
        f["key_writes"] = len(recs)
        f["key_overrides"] = len(ovs)
        f["ov_after"] = sum(1 for r in ovs if r.get("t", 0) > t)
        vals = [r.get("value") for r in sorted(recs, key=lambda r: r.get("t", 0))]
        f["alt_flips_key"] = sum(1 for i in range(2, len(vals))
                                 if vals[i] == vals[i - 2] and vals[i] != vals[i - 1])
        f["distinct_vals"] = len(set(vals))
        f["cfmsg_after"] = sum(1 for e in ctx.cfmsgs if e["t"] > t)
        f["key_cfmsgs"] = sum(1 for e in ctx.cfmsgs if key in (e.get("text") or "").lower())
        writers = {r.get("agent") for r in recs}
        f["n_writers"] = len(writers)
        f["write_after"] = sum(1 for r in recs if r.get("t", 0) > t)
        f["max_key_writes_in_run"] = max((len(v) for v in ctx.state_by_key.values()),
                                         default=0)
        out.append(f)

    # ---- goal_drift: drift msgs
    for d in ctx.drifts:
        f = common(d, "goal_drift")
        t = d["t"]
        aft = [e for e in ev if e["t"] > t]
        early = {e["sub"] for e in ev if e["sub"] and e["t"] <= t}
        f["drift_after"] = sum(1 for e in ctx.drifts if e["t"] > t)
        new_work = [e for e in aft if e["sub"] and e["sub"] not in early
                    and e["kind"] in ("work", "deliver", "assign", "ack")]
        f["new_sub_work_after"] = len(new_work)
        f["new_subs_after"] = len({e["sub"] for e in new_work})
        f["off_work_frac_after"] = (len(new_work) / len(aft)) if aft else 0
        f["offtopic_art_after"] = sum(
            1 for a in ctx.run["artifacts"]
            if a.get("subtask") == "off_topic" and a.get("t", 0) > t)
        f["any_prod_after"] = int(bool(ctx.produced_after(t)))
        f["n_prod_after"] = len(ctx.produced_after(t))
        f["drift_tail"] = int(t >= 0.8 * ctx.n)
        out.append(f)

    # ---- runaway_loop: loop series starts (first of each sender/sub key)
    series_first = {}
    for e in ctx.loops:
        key = (e["from"], e["sub"])
        series_first.setdefault(key, []).append(e)
    for (fr, s), lst in series_first.items():
        e0 = lst[0]
        f = common(e0, "runaway_loop")
        t = e0["t"]
        f["series_len"] = len(lst)
        f["series_after"] = sum(1 for x in lst if x["t"] > t)
        f["series_span"] = (lst[-1]["t"] - lst[0]["t"]) if len(lst) > 1 else 0
        f["loops_any_after"] = sum(1 for x in ctx.loops if x["t"] > t)
        f["sub_produced_later"] = int(bool([x for x in ctx.prod_times(s, fr) if x > t]))
        f["sub_produced_any_later"] = int(bool([x for x, ag, ss in ctx.prod
                                                if ss == s and x > t]))
        last_prod = max([x for x, ag, ss in ctx.prod if ss == s and ag == fr] + [-1])
        f["loops_after_prod"] = sum(1 for x in lst if x["t"] > last_prod)
        f["sub_in_assigns"] = int(s in ctx.assign_by_sub)
        out.append(f)

    return out


def build_anchor_table(rows):
    """rows: list of train rows (dicts). Returns per-run anchor arrays."""
    tables = []
    for row in rows:
        run = parse_run(row)
        ctx = RunContext(run)
        anchors = anchor_candidates(ctx)
        tables.append({"run_id": row.get("run_id"), "ctx": ctx, "anchors": anchors,
                       "label": row.get("label"), "fault_turn": row.get("fault_turn")})
    return tables
