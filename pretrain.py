#!/usr/bin/env python3
"""Offline pre-training for Coordination Referee.

Trains every model on a (battle) train.csv and saves all artifacts into a
weights/ directory that is shipped inside the submission zip next to
solution.py. solution.py then only runs inference.

    python3 pretrain.py --train data/train_battle.csv --out weights

Optional:
    --nn-seeds 1,2,3    seeds of the sequence net bagging (default 1,2,3)
    --nn-epochs 12      epochs per seed (default 12; 10-14 is fine)
    --folds 5           CV folds for offset/threshold tuning
    --skip-nn-oof       don't build NN OOF (offsets tuned on LGBM only)
    --jobs N            parallel feature extraction (default: all cores)

GPU is used automatically when torch sees CUDA. Runtime on a modern GPU
~25-35 min; CPU-only works but the NN part is slow (use --nn-seeds 1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sol"))

from engine import parse_run                      # noqa: E402
from features4 import extract_features            # noqa: E402
from features4 import wait_parts, sub_eq          # noqa: E402
from collections import defaultdict               # noqa: E402
from localize import localize                     # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
LIDX = {l: i for i, l in enumerate(LABELS)}
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def _work(rec):
    run = parse_run(rec)
    return extract_features(run)


def build_X(df, jobs):
    recs = df.to_dict("records")
    if jobs > 1:
        with Pool(jobs) as p:
            feats = p.map(_work, recs, chunksize=32)
    else:
        feats = [_work(r) for r in recs]
    cols = sorted(set().union(*[f.keys() for f in feats]))
    X = pd.DataFrame(feats).reindex(columns=cols).fillna(0.0).astype(np.float32)
    return X, recs


def fit_lgb_label(X, y, seed=42):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=7, n_estimators=1200,
        learning_rate=0.04, num_leaves=63, min_child_samples=20,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, n_jobs=-1, random_state=seed, verbose=-1)
    m.fit(X, y)
    return m


def fit_lgb_success(Xs, s, seed=42):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        objective="binary", n_estimators=800, learning_rate=0.05,
        num_leaves=63, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
        random_state=seed, verbose=-1)
    m.fit(Xs, s)
    return m


# --------------------------------------------------------------- NN parts
def nn_train_full(seqs, T, y, epochs, seed, device, time_guard=None):
    import torch
    from nn_model import SeqModel, pad_batch
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SeqModel(T.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    Tt = torch.from_numpy(np.ascontiguousarray(T.astype(np.float32)))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    n = len(seqs)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            xb, mask = pad_batch([seqs[j] for j in idx], device)
            tb = Tt[idx].to(device)
            yb = yt[idx].to(device)
            loss = lossf(model(xb, tb, mask), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        if time_guard and time.time() > time_guard:
            log("NN time guard hit at epoch %d" % ep)
            break
    return model


def nn_predict(model, seqs, T, device, bs=256):
    import torch
    from nn_model import pad_batch
    Tt = torch.from_numpy(np.ascontiguousarray(T.astype(np.float32)))
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            xb, mask = pad_batch(seqs[i:i + bs], device)
            tb = Tt[i:i + bs].to(device)
            out.append(torch.softmax(model(xb, tb, mask), -1).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--nn-seeds", default="1,2,3")
    ap.add_argument("--nn-epochs", type=int, default=12)
    ap.add_argument("--skip-nn-oof", action="store_true")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train = pd.read_csv(args.train)
    log("train %d rows, extracting features (%d jobs)" % (len(train), args.jobs))
    X, recs = build_X(train, args.jobs)
    log("features: %d" % X.shape[1])
    y = train["label"].map(LIDX).values
    s = train["success"].values.astype(int)
    from sklearn.model_selection import StratifiedKFold
    import scipy.special as sp
    from sklearn.metrics import f1_score

    # ---------------- LGBM OOF ----------------
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=42)
    oof_lgb = np.zeros((len(y), 7))
    for tr_, te in skf.split(X, y):
        m = fit_lgb_label(X.iloc[tr_], y[tr_])
        pr = m.predict_proba(X.iloc[te])
        cols = [int(c) if not isinstance(c, str) else LIDX[c] for c in m.classes_]
        oof_lgb[np.ix_(te, cols)] = pr
        log("lgb fold done")
    np.save(os.path.join(args.out, "_oof_lgb.npy"), oof_lgb)

    # ---------------- NN OOF (optional) ----------------
    oof_nn = None
    torch_ok = False
    try:
        import torch
        torch_ok = True
    except Exception as e:
        log("torch unavailable (%s) -> LGBM-only pipeline" % e)
    seeds = [int(x) for x in args.nn_seeds.split(",") if x]

    if torch_ok and not args.skip_nn_oof:
        from sklearn.preprocessing import StandardScaler
        from nn_model import build_sequence
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        log("NN OOF on %s" % dev)
        scaler = StandardScaler().fit(X.values)
        T = scaler.transform(X.values).astype(np.float32)
        log("parsing runs for sequences")
        runs = [parse_run(r) for r in recs]
        seqs = [build_sequence(r) for r in runs]
        oof_nn = np.zeros((len(y), 7))
        for fi, (tr_, te) in enumerate(skf.split(T, y)):
            model = nn_train_full([seqs[i] for i in tr_], T[tr_], y[tr_],
                                  args.nn_epochs, 100 + fi, dev)
            oof_nn[te] = nn_predict(model, [seqs[i] for i in te], T[te], dev)
            log("nn oof fold %d done" % fi)
        del model
        np.save(os.path.join(args.out, "_oof_nn.npy"), oof_nn)

    # ---------------- offsets ----------------
    def tune(bl):
        off = np.zeros(7)
        best = f1_score(y, bl.argmax(1), average="macro")
        for _ in range(3):
            for c in range(7):
                for d in (-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3):
                    t = off.copy()
                    t[c] += d
                    v = f1_score(y, (bl + t).argmax(1), average="macro")
                    if v > best:
                        best, off = v, t
        return off, best

    if oof_nn is not None:
        bl = 0.7 * sp.logit(np.clip(oof_nn, 1e-6, 1 - 1e-6)) + \
             0.3 * sp.logit(np.clip(oof_lgb, 1e-6, 1 - 1e-6))
        off_blend, mf_blend = tune(bl)
        off_lgb, mf_lgb = tune(sp.logit(np.clip(oof_lgb, 1e-6, 1 - 1e-6)))
        log("OOF blend MacroF1 %.4f (offsets %s)" %
            (mf_blend, np.round(off_blend, 2).tolist()))
        log("OOF lgb-only MacroF1 %.4f (offsets %s)" %
            (mf_lgb, np.round(off_lgb, 2).tolist()))
    else:
        bl = sp.logit(np.clip(oof_lgb, 1e-6, 1 - 1e-6))
        off_blend, mf_blend = tune(bl)
        off_lgb = off_blend
        mf_lgb = mf_blend
        log("OOF lgb-only MacroF1 %.4f (offsets %s)" %
            (mf_blend, np.round(off_blend, 2).tolist()))

    # ---------------- final models on full data ----------------
    log("final LGBM label")
    lgb_lab = fit_lgb_label(X, y)
    lgb_lab.booster_.save_model(os.path.join(args.out, "lgb_label.txt"))

    log("final LGBM success")
    lab_1h = np.zeros((len(y), 7), dtype=np.float32)
    lab_1h[np.arange(len(y)), y] = 1
    Xs = np.hstack([X.values.astype(np.float32), lab_1h])
    lgb_suc = fit_lgb_success(Xs, s)
    lgb_suc.booster_.save_model(os.path.join(args.out, "lgb_success.txt"))

    if torch_ok and not args.skip_nn:
        from sklearn.preprocessing import StandardScaler
        from nn_model import build_sequence
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        scaler = StandardScaler().fit(X.values)
        T = scaler.transform(X.values).astype(np.float32)
        log("parsing runs for final NN")
        runs = [parse_run(r) for r in recs]
        seqs = [build_sequence(r) for r in runs]
        import torch
        log("final NN: %d seeds x %d epochs on %s" %
            (len(seeds), args.nn_epochs, dev))
        for sd in seeds:
            model = nn_train_full(seqs, T, y, args.nn_epochs, sd, dev)
            torch.save(model.state_dict(),
                       os.path.join(args.out, "nn_seed%d.pt" % sd))
            log("nn seed %d saved" % sd)
            del model

    meta = {
        "labels": LABELS,
        "feature_cols": list(X.columns),
        "scaler_mean": scaler.mean_.tolist() if (torch_ok and not args.skip_nn) else None,
        "scaler_scale": scaler.scale_.tolist() if (torch_ok and not args.skip_nn) else None,
        "offsets_blend": off_blend.tolist(),
        "offsets_lgb": off_lgb.tolist(),
        "nn_seeds": seeds if (torch_ok and not args.skip_nn) else [],
        "nn_epochs": args.nn_epochs,
        "nn_weight": 0.7,
        "oof_macro_f1_blend": float(mf_blend),
        "oof_macro_f1_lgb": float(mf_lgb),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    log("saved weights to %s" % args.out)
    log("OOF MacroF1: blend %.4f | lgb %.4f" % (mf_blend, mf_lgb))


if __name__ == "__main__":
    main()
