#!/usr/bin/env python3
"""Coordination Referee — participant solution.

Contract:
    python solution.py --train train.csv --test test.csv --output predictions.csv

Pipeline
    run -> semantic events (templates mined from the simulator vocabulary)
        -> structural evidence features (handoff accounting, wait ping-pong,
           state conflicts, drift shift, loop series, run endings)
        -> LightGBM (label + success)
        -> BiGRU sequence model over events (GPU if available, CPU fallback)
        -> logit blend + macro-F1 class offsets
        -> rule-based fault_turn localisation per predicted class

Deterministic: fixed seeds everywhere. No network access required.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

T0 = time.time()

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
LIDX = {l: i for i, l in enumerate(LABELS)}
RUN_COLUMNS = ["goal", "agents", "shared_state", "messages", "artifacts", "topology"]

# macro-F1 class offsets tuned on train OOF
OFFSETS = np.array([-0.65, -0.25, -0.55, 0.15, -0.65, 0.60, 0.00])
NN_WEIGHT = 0.70          # weight of the NN logits in the blend


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def write_fallback(test, path, why):
    log("WARNING: %s -> constant predictions" % why)
    ids = test["run_id"] if "run_id" in test.columns else range(len(test))
    pd.DataFrame({"run_id": ids, "label": "clean", "success": 1,
                  "fault_turn": -1}).to_csv(path, index=False)


# ---------------------------------------------------------------- features
def build_matrices(train, test):
    from features4 import extract_features
    from engine import parse_run
    Xtr, runs_tr = [], []
    for rec in train.to_dict("records"):
        run = parse_run(rec)
        Xtr.append(extract_features(run))
        runs_tr.append(run)
    Xte, runs_te = [], []
    for rec in test.to_dict("records"):
        run = parse_run(rec)
        Xte.append(extract_features(run))
        runs_te.append(run)
    cols = sorted(set().union(*[r.keys() for r in Xtr]))
    Xtr = pd.DataFrame(Xtr).reindex(columns=cols).fillna(0.0)
    Xte = pd.DataFrame(Xte).reindex(columns=cols).fillna(0.0)
    return Xtr, Xte, runs_tr, runs_te


# ---------------------------------------------------------------- lightgbm
def fit_lgb(Xtr, y, task):
    import lightgbm as lgb
    if task == "multi":
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=7, n_estimators=1200,
            learning_rate=0.04, num_leaves=63, min_child_samples=20,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)
    else:
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=800, learning_rate=0.05,
            num_leaves=63, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
            random_state=42, verbose=-1)
    m.fit(Xtr, y)
    return m


# ---------------------------------------------------------------- sequence NN
def train_nn(runs_tr, Ttr, y, epochs, seeds, device):
    import torch
    from nn_model import SeqModel, build_sequence, pad_batch
    torch.manual_seed(42)
    np.random.seed(42)
    seqs = [build_sequence(r) for r in runs_tr]
    Ttr_t = torch.from_numpy(Ttr.astype(np.float32))
    ytr_t = torch.from_numpy(y)
    proba = np.zeros((len(y), 7), dtype=np.float64)  # test proba accumulator
    return_none = None
    models = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = SeqModel(Ttr.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        n = len(seqs)
        for ep in range(epochs):
            model.train()
            perm = np.random.permutation(n)
            for i in range(0, n, 64):
                idx = perm[i:i + 64]
                xb, mask = pad_batch([seqs[j] for j in idx], device)
                tb = Ttr_t[idx].to(device)
                yb = ytr_t[idx].to(device)
                loss = lossf(model(xb, tb, mask), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sch.step()
            if time.time() - T0 > 1380:      # hard wall-clock guard
                log("NN: time guard hit, stopping training early")
                break
        models.append(model)
    return models, seqs


def nn_predict(models, seqs_tr_stat, runs_te, Tte, device):
    import torch
    from nn_model import build_sequence, pad_batch
    seqs_te = [build_sequence(r) for r in runs_te]
    Tte_t = torch.from_numpy(Tte.astype(np.float32))
    out = np.zeros((len(runs_te), 7), dtype=np.float64)
    with torch.no_grad():
        for model in models:
            model.eval()
            preds = []
            for i in range(0, len(seqs_te), 256):
                xb, mask = pad_batch(seqs_te[i:i + 256], device)
                tb = Tte_t[i:i + 256].to(device)
                preds.append(torch.softmax(model(xb, tb, mask), -1).cpu().numpy())
            out += np.concatenate(preds)
    return out / max(1, len(models))


# ---------------------------------------------------------------- fault turn
from features4 import wait_parts, sub_eq  # noqa: E402
from engine import events_of  # noqa: E402


def productions(run):
    prod, prod_t = set(), []
    for e in run["events"]:
        if e["kind"] == "deliver" and e["sub"]:
            prod.add(e["sub"])
            prod_t.append((e["t"], e["from"], e["sub"]))
    for a in run["artifacts"]:
        if a.get("subtask") and a.get("status") != "copy":
            prod.add(a["subtask"])
            prod_t.append((a.get("t", 0), a.get("by"), a["subtask"]))
    return prod, prod_t


def localize(run, label):
    ev = run["events"]
    n = len(ev)
    if not ev or label == "clean":
        return -1
    prod, prod_t = productions(run)
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
        early_full = None
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
            def nrm(t):
                t = (t or "").lower()
                t = re.sub(r"art_\S+|a\d+", "@", t)
                t = re.sub(r"\d+", "#", t)
                return re.sub(r"\s+", " ", t).strip()
            by_key = defaultdict(list)
            for e in loops:
                by_key[(e["from"], e["sub"] or nrm(e["text"]))].append(e)
            best = max(by_key.values(), key=len)
            return int(best[0]["t"])
        return int(ev[len(ev) // 2]["t"])

    return -1


import re  # noqa: E402
from collections import defaultdict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    log("train=%d test=%d" % (len(train), len(test)))

    if any(c not in test.columns for c in RUN_COLUMNS) or \
            any(c not in train.columns for c in RUN_COLUMNS + ["label", "success"]):
        write_fallback(test, args.output, "input is not a full run table")
        return

    log("extracting features")
    Xtr, Xte, runs_tr, runs_te = build_matrices(train, test)
    log("features: %d" % Xtr.shape[1])

    y = train["label"].map(LIDX).values
    s = train["success"].values.astype(int)

    # ---------------- lightgbm label ----------------
    log("fitting LGBM label")
    lgb_lab = fit_lgb(Xtr, y, "multi")
    pr = lgb_lab.predict_proba(Xte)
    cols = [int(c) if not isinstance(c, str) else LIDX[c] for c in lgb_lab.classes_]
    p_lgb = np.zeros((len(Xte), 7))
    p_lgb[:, cols] = pr
    p_lgb_tr = None

    # ---------------- sequence NN ----------------
    p_nn = None
    try:
        import torch
        from sklearn.preprocessing import StandardScaler
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        log("sequence model on %s" % dev)
        scaler = StandardScaler().fit(Xtr.values)
        Ttr = scaler.transform(Xtr.values)
        Tte = scaler.transform(Xte.values)
        epochs, seeds = (12, [1, 2, 3]) if dev == "cuda" else (8, [1])
        if dev != "cuda":
            epochs = 6
        models, _ = train_nn(runs_tr, Ttr, y, epochs, seeds, dev)
        p_nn = nn_predict(models, None, runs_te, Tte, dev)
        del models
    except Exception as exc:
        log("NN failed (%s) -> LGBM only" % exc)

    # ---------------- blend ----------------
    import scipy.special as sp
    if p_nn is not None:
        logits = NN_WEIGHT * sp.logit(np.clip(p_nn, 1e-6, 1 - 1e-6)) + \
                 (1 - NN_WEIGHT) * sp.logit(np.clip(p_lgb, 1e-6, 1 - 1e-6))
    else:
        logits = sp.logit(np.clip(p_lgb, 1e-6, 1 - 1e-6))
    pred_idx = (logits + OFFSETS).argmax(1)
    pred_label = [LABELS[i] for i in pred_idx]
    log("labels predicted")

    # ---------------- success ----------------
    log("fitting success model")
    lab_1h = np.zeros((len(y), 7))
    lab_1h[np.arange(len(y)), y] = 1
    Xs = np.hstack([Xtr.values, lab_1h])
    suc = fit_lgb(pd.DataFrame(Xs), s, "binary")
    lab_1h_te = np.zeros((len(Xte), 7))
    lab_1h_te[np.arange(len(Xte)), pred_idx] = 1
    Xse = np.hstack([Xte.values, lab_1h_te])
    p_suc = suc.predict_proba(Xse)[:, 1]
    pred_success = (p_suc > 0.5).astype(int)

    # ---------------- fault turn ----------------
    log("localising faults")
    pred_turn = [localize(run, lab) for run, lab in zip(runs_te, pred_label)]

    out = pd.DataFrame({"run_id": test["run_id"], "label": pred_label,
                        "success": pred_success, "fault_turn": pred_turn})
    out.to_csv(args.output, index=False)
    log("wrote %s (%d rows)" % (args.output, len(out)))


if __name__ == "__main__":
    main()
