"""Rich run-level feature extraction for Coordination Referee.

Builds on engine.py semantic events. All features are structural (robust to
vocabulary shift) or normalized by run length / agent count.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from engine import parse_run, events_of

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]


def _norm_text(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"art_\S+", "@", t)
    t = re.sub(r"\ba\d+\b", "@", t)
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"[^a-z@# ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _safe(a, b):
    return float(a) / b if b else 0.0


def _entropy(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log(c / total + 1e-12) for c in counts if c > 0)


def extract_run_features(run) -> dict:
    f = {}
    ev = run["events"]
    n = len(ev)
    agents = run["agents"]
    arts = run["artifacts"]
    state = run["shared_state"]
    n_ag = max(1, len(agents))
    kinds = Counter(e["kind"] for e in ev)

    roles = {}
    for a in agents:
        roles[a.get("id")] = a.get("role", "")
    orch = next((aid for aid, r in roles.items() if r == "orchestrator"), None)

    # ---------------- volume ----------------
    f["n_msg"] = n
    f["n_agents"] = len(agents)
    f["n_art"] = len(arts)
    f["n_state"] = len(state)
    f["msg_per_agent"] = _safe(n, len(agents))
    for k in ("assign", "deliver", "ack", "work", "wait", "loop", "drift",
              "conflict_msg", "state_write", "recovery", "final", "system"):
        f[f"n_{k}"] = kinds.get(k, 0)
        f[f"sh_{k}"] = _safe(kinds.get(k, 0), n)

    # senders
    senders = Counter(e["from"] for e in ev)
    f["n_senders"] = len(senders)
    f["sender_entropy"] = _entropy(list(senders.values()))
    f["sender_max_share"] = _safe(max(senders.values()) if senders else 0, n)

    # ---------------- subtasks / plan ----------------
    assigns = [e for e in ev if e["kind"] in ("assign", "assign_maybe") and e["sub"]]
    delivers = [e for e in ev if e["kind"] == "deliver" and e["sub"]]
    sub_all = {e["sub"] for e in ev if e["sub"]}
    assign_by_sub = defaultdict(list)
    for a in assigns:
        assign_by_sub[a["sub"]].append(a)
    # delivery times per (sub, agent): from deliver msgs and non-copy artifacts
    prod_times = defaultdict(list)   # sub -> [(t, agent)]
    for e in delivers:
        prod_times[e["sub"]].append((e["t"], e["from"]))
    for a in arts:
        if a.get("subtask") and a.get("status") != "copy":
            prod_times[a["subtask"]].append((a.get("t", 0), a.get("by")))
    prod_agents = defaultdict(set)
    for s, lst in prod_times.items():
        for _, ag in lst:
            prod_agents[s].add(ag)

    f["n_subs"] = len(sub_all)
    f["n_assign_subs"] = len(assign_by_sub)
    f["n_prod_subs"] = len(prod_agents)
    f["coverage"] = _safe(sum(1 for s in assign_by_sub if prod_agents.get(s)),
                          len(assign_by_sub))
    f["subs_2plus_arts"] = sum(1 for s, lst in prod_times.items()
                               if len({ag for _, ag in lst}) >= 2)
    # artifacts
    hashes = Counter(a.get("hash") for a in arts if a.get("status") != "copy")
    f["dup_hash_pairs"] = sum(v - 1 for v in hashes.values() if v > 1)
    st_counts = Counter(a.get("status") for a in arts)
    f["art_copy"] = st_counts.get("copy", 0)
    f["art_partial"] = st_counts.get("partial", 0)
    f["has_final_delivery_art"] = int(any(a.get("subtask") == "final_delivery" for a in arts))
    fd = [a for a in arts if a.get("subtask") == "final_delivery"]
    f["final_art_partial"] = int(bool(fd) and fd[0].get("status") == "partial")
    # off-topic artifacts
    f["n_offtopic_art"] = sum(1 for a in arts if a.get("subtask") == "off_topic")

    # per-assignee accounting
    undelivered = 0; undel_silent = 0; undel_active = 0
    reassign_same_from = 0; reassign_diff_from = 0
    dh_cands = []
    for s, lst in assign_by_sub.items():
        for a in lst:
            prods = prod_agents.get(s, set())
            if a["to"] not in prods and s not in {p for p in prods}:
                pass
        producers = prod_agents.get(s, set())
        first = lst[0]
        if not (producers & {x["to"] for x in lst}):
            undelivered += 1
            to = first["to"]
            after = [e for e in ev if e["t"] > first["t"] and e["from"] == to]
            if after:
                undel_active += 1
            else:
                undel_silent += 1
            dh_cands.append(first["t"])
        # same-from reassigns
        froms = [x["from"] for x in lst]
        for i in range(1, len(lst)):
            if froms[i] == froms[i - 1]:
                reassign_same_from += 1
            else:
                reassign_diff_from += 1
    f["n_undel_sub"] = undelivered
    f["n_undel_silent"] = undel_silent
    f["n_undel_active"] = undel_active
    f["undel_ratio"] = _safe(undelivered, len(assign_by_sub))
    f["n_reassign_same"] = reassign_same_from
    f["n_reassign_diff"] = reassign_diff_from

    # duplicated-work events (strong): same-from reassign + both produced
    dw_events = 0; dw_after_deliv = 0; dw_same_hash = 0
    art_hash = {}
    for a in arts:
        if a.get("subtask") and a.get("status") != "copy":
            art_hash[(a.get("by"), a["subtask"])] = a.get("hash")
    for s, lst in assign_by_sub.items():
        for i in range(1, len(lst)):
            a0, a1 = lst[i - 1], lst[i]
            if a0["from"] != a1["from"]:
                continue
            prods = {a0["to"], a1["to"]} & prod_agents.get(s, set())
            if len(prods) < 2:
                continue
            dw_events += 1
            d1 = [t for t, ag in prod_times.get(s, []) if ag == a0["to"] and t < a1["t"]]
            if d1:
                dw_after_deliv += 1
            h0, h1 = art_hash.get((a0["to"], s)), art_hash.get((a1["to"], s))
            if h0 is not None and h0 == h1:
                dw_same_hash += 1
    f["n_dw_events"] = dw_events
    f["n_dw_after_deliv"] = dw_after_deliv
    f["n_dw_same_hash"] = dw_same_hash

    # ---------------- waits / deadlock ----------------
    waits = events_of(run, "wait")
    last_prod_t = max([t for lst in prod_times.values() for t, _ in lst] + [-1])
    pair_waits = defaultdict(list)
    for w in waits:
        pair_waits[(w["from"], w["to"])].append(w)
    mutual_pairs = 0; unres_pairs = 0; unres_first_t = []
    for (a, b), ws in pair_waits.items():
        if a < b and pair_waits.get((b, a)):
            allw = sorted(ws + pair_waits[(b, a)], key=lambda x: x["t"])
            mutual_pairs += 1
            first_t = allw[0]["t"]
            # resolved if any production of a waited sub after first_t by target
            resolved = False
            for w in allw:
                awaited = w["sub"]
                if any(ag == w["to"] and t > w["t"] for t, ag in prod_times.get(awaited, [])):
                    resolved = True
                    break
            if not resolved:
                unres_pairs += 1
                unres_first_t.append(first_t)
    f["n_mutual_pairs"] = mutual_pairs
    f["n_unres_pairs"] = unres_pairs
    f["n_waits_after_last_prod"] = sum(1 for w in waits if w["t"] > last_prod_t)
    tail0 = int(n * 0.8)
    f["n_waits_tail"] = sum(1 for w in waits if w["t"] >= tail0)
    f["sh_waits_tail"] = _safe(f["n_waits_tail"], max(1, n - tail0))
    # ping-pong: max min-count over ordered pairs
    pp = max([min(len(pair_waits[(a, b)]), len(pair_waits[(b, a)]))
              for (a, b) in pair_waits if a < b and (b, a) in pair_waits] or [0])
    f["pingpong"] = pp

    # ---------------- conflict / state ----------------
    per_key = defaultdict(list)
    for s in state:
        if s.get("key") == "run_status":
            continue
        per_key[s["key"]].append(s)
    overrides = 0; hot_ov = 0; alt_flips = 0
    for k, recs in per_key.items():
        ovs = [r for r in recs if r.get("op") == "override"]
        overrides += len(ovs)
        hot_ov = max(hot_ov, len(ovs))
        vals = [r.get("value") for r in recs]
        alt_flips += sum(1 for i in range(2, len(vals))
                         if vals[i] == vals[i - 2] and vals[i] != vals[i - 1])
    f["n_overrides"] = overrides
    f["hot_key_overrides"] = hot_ov
    f["state_alt_flips"] = alt_flips
    f["state_keys"] = len(per_key)
    f["state_hotkey_writes"] = max((len(v) for v in per_key.values()), default=0)
    f["cfmsg_per_override"] = _safe(kinds.get("conflict_msg", 0), overrides)
    tail0 = int(len(state) * 0.8)
    f["n_overrides_tail"] = sum(1 for s in state
                                if s.get("op") == "override" and s.get("t", 0) >= tail0)

    # ---------------- drift / goal ----------------
    drifts = events_of(run, "drift")
    goal_tok = set(re.findall(r"[a-z]{3,}", (run.get("goal") or "").lower()))
    f["n_drift"] = len(drifts)
    first_drift_t = drifts[0]["t"] if drifts else 10 ** 9
    # new subtasks after first drift
    early = {e["sub"] for e in ev if e["sub"] and e["t"] < first_drift_t}
    late = {e["sub"] for e in ev if e["sub"] and e["t"] >= first_drift_t}
    f["new_subs_after_drift"] = len(late - early) if drifts else 0
    f["work_after_drift_offtopic"] = sum(
        1 for e in ev if e["t"] > first_drift_t and e["kind"] in ("work", "deliver")
        and e["sub"] and e["sub"] not in early) if drifts else 0
    f["drift_tail"] = sum(1 for d in drifts if d["t"] >= tail0)
    # goal token overlap head vs tail
    texts = [e["text"] for e in ev]
    if texts:
        k = max(1, n // 4)
        ovs_ = [_safe(len(goal_tok & set(re.findall(r"[a-z]{3,}", t.lower()))), len(goal_tok) or 1)
                for t in texts]
        f["goal_ov_head"] = sum(ovs_[:k]) / k
        f["goal_ov_tail"] = sum(ovs_[-k:]) / k
        f["goal_ov_delta"] = f["goal_ov_tail"] - f["goal_ov_head"]
        f["goal_ov_zero_tail"] = _safe(sum(1 for o in ovs_[-k:] if o == 0), k)

    # ---------------- loop ----------------
    loops = events_of(run, "loop")
    by_key = defaultdict(list)
    for e in loops:
        by_key[(e["from"], e["sub"] or _norm_text(e["text"]))].append(e)
    f["n_loop"] = len(loops)
    f["loop_max_rep"] = max((len(v) for v in by_key.values()), default=0)
    f["loop_keys_3plus"] = sum(1 for v in by_key.values() if len(v) >= 3)
    f["loop_tail"] = sum(1 for e in loops if e["t"] >= tail0)
    # looped subtasks that never got produced
    f["loop_subs_unprod"] = sum(1 for (fr, s) in by_key
                                if len(by_key[(fr, s)]) >= 3 and s in sub_all
                                and not prod_agents.get(s))
    # max identical (sender, normtext) streak & count
    cnt = Counter((e["from"], _norm_text(e["text"])) for e in ev)
    f["max_identical_sender_text"] = max(cnt.values()) if cnt else 0
    # consecutive identical norm streak
    best = cur = 0; prev = None
    for e in ev:
        x = _norm_text(e["text"])
        cur = cur + 1 if x == prev else 1
        best = max(best, cur); prev = x
    f["max_consec_identical"] = best

    # ---------------- run end / status ----------------
    f["has_final_msg"] = kinds.get("final", 0) > 0
    f["has_system_msg"] = kinds.get("system", 0) > 0
    sys_texts = [e["text"].lower() for e in ev if e["kind"] == "system"]
    f["sys_no_progress"] = int(any("no progress" in t for t in sys_texts))
    f["sys_budget"] = int(any("budget" in t or "terminated" in t for t in sys_texts))
    f["sys_halting"] = int(any("halting" in t or "stopping" in t for t in sys_texts)
                           and not any("no progress" in t or "budget" in t for t in sys_texts))
    st = [s for s in state if s.get("key") == "run_status"]
    v = st[-1].get("value") if st else "none"
    for val in ("completed", "timeout", "aborted", "incomplete", "none"):
        f[f"rs_{val}"] = int(v == val)
    # last message index of production vs end
    f["prod_end_frac"] = _safe(last_prod_t, max(1, n - 1))
    f["tail_quiet"] = _safe(n - 1 - last_prod_t, max(1, n))

    # ---------------- topology ----------------
    tt = run.get("topo_type", "")
    for t in ("star", "pipeline", "mesh", "hierarchical", "blackboard"):
        f[f"topo_{t}"] = float(tt == t)
    f["topo_edges"] = len(run.get("topo_edges", []))

    # actual communication graph vs allowed
    allowed = {frozenset(e) for e in run.get("topo_edges", []) if len(e) == 2}
    viol = 0; pairs = set()
    for e in ev:
        if e["from"] and e["to"] and e["from"] != e["to"]:
            pairs.add(frozenset((e["from"], e["to"])))
            if allowed and frozenset((e["from"], e["to"])) not in allowed:
                viol += 1
    f["edge_viol"] = viol
    f["edge_viol_share"] = _safe(viol, len(pairs) or 1)
    return f


def featurize_row(row: dict):
    run = parse_run(row)
    return run, extract_run_features(run)
