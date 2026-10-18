#!/usr/bin/env python3
"""Coordination Referee — participant solution.

Contract:
    python solution.py --train train.csv --test test.csv --output predictions.csv

Three execution modes, chosen automatically:

1. weights/   -> pre-trained models shipped in this archive: LightGBM label
                 + LightGBM success + bagged BiGRU sequence nets. Only
                 inference runs here (a couple of minutes).
2. no weights -> quick in-job training of the LightGBM heads on the provided
                 train.csv (few minutes), same feature pipeline.
3. smoke test -> test.csv lacks the run columns (only submission columns):
                 write a valid constant predictions.csv and exit 0.

Deterministic: fixed seeds everywhere; no network access needed.
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
sys.path.insert(0, os.path.join(HERE, "sol"))

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
LIDX = {l: i for i, l in enumerate(LABELS)}
RUN_COLUMNS = ["goal", "agents", "shared_state", "messages", "artifacts", "topology"]
DEFAULT_OFFSETS_LGB = np.array([-0.60, -0.20, -0.50, 0.10, -0.60, 0.55, 0.00])

T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def write_fallback(test, path, why):
    log("WARNING: %s -> constant predictions" % why)
    ids = test["run_id"] if "run_id" in test.columns else range(len(test))
    pd.DataFrame({"run_id": ids, "label": "clean", "success": 1,
                  "fault_turn": -1}).to_csv(path, index=False)


# --------------------------------------------------------------- features
def build_matrix(df, jobs=1):
    from features4 import extract_features
    from engine import parse_run
    from multiprocessing import Pool

    recs = df.to_dict("records")

    def _work(rec):
        run = parse_run(rec)
        return extract_features(run)

    if jobs > 1 and len(recs) > 400:
        with Pool(jobs) as p:
            feats = p.map(_work, recs, chunksize=32)
    else:
        feats = [_work(r) for r in recs]
    cols = sorted(set().union(*[f.keys() for f in feats]))
    X = pd.DataFrame(feats).reindex(columns=cols).fillna(0.0).astype(np.float32)
    return X, recs


def parse_runs(recs):
    from engine import parse_run
    return [parse_run(r) for r in recs]


def reindex_like(X, cols):
    return X.reindex(columns=cols).fillna(0.0).astype(np.float32)


# --------------------------------------------------------------- in-job training
def quick_train(X, y, s):
    import lightgbm as lgb
    lab = lgb.LGBMClassifier(
        objective="multiclass", num_class=7, n_estimators=1200,
        learning_rate=0.04, num_leaves=63, min_child_samples=20,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)
    lab.fit(X, y)
    lab_1h = np.zeros((len(y), 7), dtype=np.float32)
    lab_1h[np.arange(len(y)), y] = 1
    Xs = np.hstack([X.values.astype(np.float32), lab_1h])
    suc = lgb.LGBMClassifier(
        objective="binary", n_estimators=800, learning_rate=0.05,
        num_leaves=63, subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)
    suc.fit(Xs, s)
    return lab, suc


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    test = pd.read_csv(args.test)
    log("test=%d rows" % len(test))

    if any(c not in test.columns for c in RUN_COLUMNS) or "run_id" not in test.columns:
        write_fallback(test, args.output, "test.csv has no run fields (smoke mode)")
        return

    # ------------------------------------------------- weights
    wdir = os.path.join(HERE, "weights")
    meta = None
    if os.path.exists(os.path.join(wdir, "meta.json")):
        try:
            with open(os.path.join(wdir, "meta.json")) as f:
                meta = json.load(f)
            log("weights found: %s" % wdir)
        except Exception as e:
            log("meta.json unreadable (%s) -> in-job training" % e)

    feat_cols = (meta or {}).get("feature_cols")
    has_lgb_lab = meta is not None and os.path.exists(os.path.join(wdir, "lgb_label.txt"))
    has_lgb_suc = meta is not None and os.path.exists(os.path.join(wdir, "lgb_success.txt"))
    nn_seeds = (meta or {}).get("nn_seeds") or []
    has_nn = bool(nn_seeds) and feat_cols is not None and \
        all(os.path.exists(os.path.join(wdir, "nn_seed%d.pt" % sd)) for sd in nn_seeds)
    if has_nn:
        try:
            import torch  # noqa: F401
        except Exception:
            log("torch not available -> NN disabled")
            has_nn = False

    need_train = not (has_lgb_lab and has_lgb_suc)
    train = None
    if need_train:
        train = pd.read_csv(args.train)
        if any(c not in train.columns for c in RUN_COLUMNS) or \
                "label" not in train.columns or "success" not in train.columns:
            write_fallback(test, args.output, "no usable train and no weights")
            return

    # ------------------------------------------------- test features
    log("extracting test features (%d jobs)" % args.jobs)
    Xte, recs_te = build_matrix(test, args.jobs)
    runs_te = parse_runs(recs_te)
    if feat_cols:
        Xte = reindex_like(Xte, feat_cols)
    log("test features: %d" % Xte.shape[1])

    # ------------------------------------------------- label
    p_lgb = None
    if has_lgb_lab:
        import lightgbm as lgb
        log("LGBM label: loading booster")
        booster = lgb.Booster(model_file=os.path.join(wdir, "lgb_label.txt"))
        p_lgb = np.asarray(booster.predict(Xte.values), dtype=np.float64)
        if p_lgb.ndim == 1:
            p_lgb = p_lgb.reshape(-1, 7)
    else:
        log("LGBM label: quick in-job training")
        y = train["label"].map(LIDX).values
        s = train["success"].values.astype(int)
        Xtr, _ = build_matrix(train, args.jobs)
        if feat_cols:
            Xtr = reindex_like(Xtr, feat_cols)
        lab, suc_injob = quick_train(Xtr, y, s)
        pr = lab.predict_proba(Xte.values)
        cols = [int(c) if not isinstance(c, str) else LIDX[c] for c in lab.classes_]
        p_lgb = np.zeros((len(Xte), 7))
        p_lgb[:, cols] = pr
        if not has_lgb_suc:
            globals()["_suc_injob"] = suc_injob

    # ------------------------------------------------- NN
    p_nn = None
    if has_nn:
        try:
            import torch
            from sklearn.preprocessing import StandardScaler
            from nn_model import build_sequence, SeqModel, pad_batch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            mean = np.asarray(meta["scaler_mean"], dtype=np.float32)
            scale = np.asarray(meta["scaler_scale"], dtype=np.float32)
            Tte = ((Xte.values - mean) / np.maximum(scale, 1e-8)).astype(np.float32)
            seqs_te = [build_sequence(r) for r in runs_te]
            log("NN inference: %d seeds on %s" % (len(nn_seeds), dev))
            acc = np.zeros((len(Xte), 7), dtype=np.float64)
            for sd in nn_seeds:
                model = SeqModel(Tte.shape[1]).to(dev)
                model.load_state_dict(torch.load(
                    os.path.join(wdir, "nn_seed%d.pt" % sd),
                    map_location=dev))
                model.eval()
                Tt = torch.from_numpy(Tte)
                with torch.no_grad():
                    for i in range(0, len(seqs_te), 256):
                        xb, mask = pad_batch(seqs_te[i:i + 256], dev)
                        tb = Tt[i:i + 256].to(dev)
                        acc[i:i + 256] += torch.softmax(
                            model(xb, tb, mask), -1).cpu().numpy()
                del model
            p_nn = acc / len(nn_seeds)
        except Exception as e:
            log("NN inference failed (%s) -> continuing without it" % e)
            p_nn = None

    # ------------------------------------------------- blend
    import scipy.special as sp
    nn_w = float((meta or {}).get("nn_weight", 0.7))
    if p_nn is not None and p_lgb is not None:
        logits = nn_w * sp.logit(np.clip(p_nn, 1e-6, 1 - 1e-6)) + \
                 (1 - nn_w) * sp.logit(np.clip(p_lgb, 1e-6, 1 - 1e-6))
        offsets = np.asarray(meta.get("offsets_blend",
                                      meta.get("offsets_lgb", DEFAULT_OFFSETS_LGB)))
    elif p_nn is not None:
        logits = sp.logit(np.clip(p_nn, 1e-6, 1 - 1e-6))
        offsets = np.asarray(meta.get("offsets_blend", DEFAULT_OFFSETS_LGB))
    else:
        logits = sp.logit(np.clip(p_lgb, 1e-6, 1 - 1e-6))
        src = meta.get("offsets_lgb", DEFAULT_OFFSETS_LGB) if meta else DEFAULT_OFFSETS_LGB
        if p_nn is None and has_nn is False and meta is None:
            src = DEFAULT_OFFSETS_LGB
        offsets = np.asarray(src)
    pred_idx = (logits + offsets).argmax(1)
    pred_label = [LABELS[i] for i in pred_idx]
    log("labels predicted")

    # ------------------------------------------------- success
    if has_lgb_suc:
        import lightgbm as lgb
        log("LGBM success: loading booster")
        booster = lgb.Booster(model_file=os.path.join(wdir, "lgb_success.txt"))
        lab_1h_te = np.zeros((len(Xte), 7), dtype=np.float32)
        lab_1h_te[np.arange(len(Xte)), pred_idx] = 1
        Xse = np.hstack([Xte.values.astype(np.float32), lab_1h_te])
        p_suc = np.asarray(booster.predict(Xse), dtype=np.float64).ravel()
    else:
        log("LGBM success: in-job model")
        if train is None:
            train = pd.read_csv(args.train)
        suc = globals().get("_suc_injob")
        if suc is None:
            # weights had label but not success: quick fit on provided train
            y = train["label"].map(LIDX).values
            s = train["success"].values.astype(int)
            Xtr = globals().get("_Xtr")
            if Xtr is None:
                Xtr, _ = build_matrix(train, args.jobs)
                if feat_cols:
                    Xtr = reindex_like(Xtr, feat_cols)
            lab_1h = np.zeros((len(y), 7), dtype=np.float32)
            lab_1h[np.arange(len(y)), y] = 1
            import lightgbm as lgb
            suc = lgb.LGBMClassifier(
                objective="binary", n_estimators=800, learning_rate=0.05,
                num_leaves=63, subsample=0.9, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                random_state=42, verbose=-1)
            suc.fit(np.hstack([Xtr.values.astype(np.float32), lab_1h]), s)
        lab_1h_te = np.zeros((len(Xte), 7), dtype=np.float32)
        lab_1h_te[np.arange(len(Xte)), pred_idx] = 1
        Xse = np.hstack([Xte.values.astype(np.float32), lab_1h_te])
        p_suc = suc.predict_proba(Xse)[:, 1]
    pred_success = (p_suc > 0.5).astype(int)

    # ------------------------------------------------- fault turn
    log("localising faults")
    from localize import localize
    pred_turn = [localize(run, lab) for run, lab in zip(runs_te, pred_label)]

    out = pd.DataFrame({"run_id": test["run_id"], "label": pred_label,
                        "success": pred_success, "fault_turn": pred_turn})
    out.to_csv(args.output, index=False)
    log("wrote %s (%d rows)" % (args.output, len(out)))


if __name__ == "__main__":
    main()
