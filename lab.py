#!/usr/bin/env python3
"""Локальная лаборатория Coordination Referee.

Команды:
    python lab.py score            # обучить на 85% train, посчитать скор на 15% holdout
    python lab.py tune             # подобрать бленд/пороги/смещения -> скор растёт
    python lab.py final [--retrain]# собрать my_submission.zip с лучшими настройками
    python lab.py score --seed 7   # повторить на другом holdout (проверка стабильности)

Флаги score: --gpu (бэггинг NN 3 сида x 10 эпох на CUDA), --no-nn (только LGBM, быстро),
--holdout 0.15, --jobs N.

История прогонов пишется в scores_history.csv — видно, как скор растёт от запуска
к запуску. Локальный скор — компас; финальный судья — платформа.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sol"))

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
LIDX = {l: i for i, l in enumerate(LABELS)}
T0 = time.time()
CACHE = os.path.join(HERE, "lab_cache")
HISTORY = os.path.join(HERE, "scores_history.csv")


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


# ----------------------------------------------------------------- metrics
def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=list(range(7)), average="macro"))


def binary_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p))


def hit_at_2(y, p, ft_true, ft_pred):
    y, p = np.asarray(y), np.asarray(p)
    m = (y == p) & (y != 0)
    if m.sum() == 0:
        return 0.0
    d = np.abs(np.asarray(ft_true)[m] - np.asarray(ft_pred)[m]) <= 2
    return float(d.mean())


def composite(mf1, rf1, sf1, h2):
    return 0.50 * mf1 + 0.25 * rf1 + 0.15 * sf1 + 0.10 * h2


def full_report(y, p, s, ps, na, ft_true, ft_pred):
    y, p = np.asarray(y), np.asarray(p)
    mask = np.asarray(na) >= 7
    mf = macro_f1(y, p)
    rf = macro_f1(y[mask], p[mask]) if mask.any() else 0.0
    sf = binary_f1(np.asarray(s), np.asarray(ps))
    h2 = hit_at_2(y, p, ft_true, ft_pred)
    return {"macro_f1": mf, "robustness_f1": rf, "success_f1": sf,
            "fault_turn_hit2": h2, "score": composite(mf, rf, sf, h2)}


def print_report(r, title=""):
    if title:
        print("\n===== %s =====" % title)
    print("MacroF1        %.4f" % r["macro_f1"])
    print("RobustnessF1   %.4f  (срез agents>=7, прокси сдвинутой трети)" % r["robustness_f1"])
    print("SuccessF1      %.4f" % r["success_f1"])
    print("fault_turn@2   %.4f" % r["fault_turn_hit2"])
    print("SCORE          %.4f   (= .5*MF1 + .25*RF1 + .15*SF1 + .10*hit@2)" % r["score"])


# ----------------------------------------------------------------- data
def ensure_data():
    d = os.path.join(HERE, "data")
    tr, te = os.path.join(d, "train.csv"), os.path.join(d, "test.csv")
    if not os.path.exists(tr):
        if os.path.exists(tr + ".gz"):
            log("распаковываю data/train.csv.gz")
            with gzip.open(tr + ".gz", "rb") as f, open(tr, "wb") as o:
                shutil.copyfileobj(f, o)
        else:
            # попробовать вытащить из participant-архива рядом
            cands = []
            for root in (HERE, os.path.dirname(HERE)):
                if os.path.isdir(root):
                    cands += [os.path.join(root, f) for f in os.listdir(root)
                              if f.lower().endswith(".zip")]
            for z in cands:
                try:
                    with zipfile.ZipFile(z) as zz:
                        for name in zz.namelist():
                            if name.endswith("data/train.csv"):
                                log("извлекаю train.csv из %s" % os.path.basename(z))
                                zz.extract(name, d)
                                src = os.path.join(d, name)
                                shutil.move(src, tr)
                                break
                        if os.path.exists(tr):
                            break
                except Exception:
                    continue
    if not os.path.exists(tr):
        raise SystemExit("Нет data/train.csv. Положи файл в data/ рядом с lab.py.")
    if not os.path.exists(te) and os.path.exists(te + ".gz"):
        with gzip.open(te + ".gz", "rb") as f, open(te, "wb") as o:
            shutil.copyfileobj(f, o)
    return tr, te


def ensure_weights():
    """Скопировать поставляемые веса в weights/ (если ещё нет)."""
    src = os.path.join(HERE, "weights")
    if not os.path.exists(os.path.join(src, "meta.json")):
        raise SystemExit("Нет weights/meta.json — весов нет в комплекте.")
    return src


# ----------------------------------------------------------------- shared
def load_meta(wdir):
    with open(os.path.join(wdir, "meta.json")) as f:
        return json.load(f)


def save_meta(wdir, meta):
    with open(os.path.join(wdir, "meta.json"), "w") as f:
        json.dump(meta, f)


def build_success_probs(X, pred_idx, wdir):
    import lightgbm as lgb
    with open(os.path.join(wdir, "lgb_success.txt"), "r", encoding="utf-8") as f:
        booster = lgb.Booster(model_str=f.read())
    lab_1h = np.zeros((len(X), 7), dtype=np.float32)
    lab_1h[np.arange(len(X)), pred_idx] = 1
    Xse = np.hstack([X.astype(np.float32), lab_1h])
    return np.asarray(booster.predict(Xse), dtype=np.float64).ravel()


def localize_all(runs, labels):
    from localize import localize
    return np.array([localize(r, LABELS[l]) for r, l in zip(runs, labels)], dtype=int)


# ----------------------------------------------------------------- score
def cmd_score(args):
    from pretrain import build_X, fit_lgb_label, fit_lgb_success, nn_train_full, nn_predict
    from engine import parse_run
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    os.makedirs(CACHE, exist_ok=True)
    tr_path, _ = ensure_data()
    wdir = ensure_weights()
    df = pd.read_csv(tr_path)
    log("train %d rows" % len(df))

    strat = df["label"].astype(str) + "_" + df["success"].astype(str)
    tr, ho = train_test_split(df, test_size=args.holdout, random_state=args.seed,
                              stratify=strat)
    log("split: train=%d holdout=%d (seed %d)" % (len(tr), len(ho), args.seed))

    Xtr, recs_tr = build_X(tr, args.jobs)
    Xho, recs_ho = build_X(ho, args.jobs)
    meta = load_meta(wdir)
    feat_cols = meta["feature_cols"]
    Xtr = Xtr.reindex(columns=feat_cols).fillna(0.0).astype(np.float32)
    Xho = Xho.reindex(columns=feat_cols).fillna(0.0).astype(np.float32)
    runs_ho = [parse_run(r) for r in recs_ho]

    y = tr["label"].map(LIDX).values
    yho = ho["label"].map(LIDX).values
    s = tr["success"].values.astype(int)
    sho = ho["success"].values.astype(int)
    na = ho["agents"].map(lambda a: len(json.loads(a))).values

    log("LGBM label")
    lgb_lab = fit_lgb_label(Xtr, y)
    p_lgb_ho = np.asarray(lgb_lab.predict_proba(Xho.values), dtype=np.float64)
    log("LGBM success")
    lab_1h = np.zeros((len(y), 7), dtype=np.float32)
    lab_1h[np.arange(len(y)), y] = 1
    lgb_suc = fit_lgb_success(np.hstack([Xtr.values.astype(np.float32), lab_1h]), s)

    p_nn_ho = None
    if not args.no_nn:
        try:
            import torch
            from nn_model import build_sequence
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            seeds = [1, 2, 3] if (dev == "cuda" or args.gpu) else [1]
            ep = 10 if (dev == "cuda" or args.gpu) else 6
            log("NN: %d сид x %d эпох на %s" % (len(seeds), ep, dev))
            scaler = StandardScaler().fit(Xtr.values)
            Ttr = scaler.transform(Xtr.values).astype(np.float32)
            Tho = scaler.transform(Xho.values).astype(np.float32)
            seqs_tr = [build_sequence(r) for r in [parse_run(r) for r in recs_tr]]
            seqs_ho = [build_sequence(r) for r in runs_ho]
            p_nn_ho = np.zeros((len(yho), 7))
            for sd in seeds:
                m = nn_train_full(seqs_tr, Ttr, y, ep, sd, dev)
                p_nn_ho += nn_predict(m, seqs_ho, Tho, dev)
                del m
            p_nn_ho /= len(seeds)
        except Exception as e:
            log("NN недоступен (%s) -> только LGBM" % e)
            p_nn_ho = None

    np.save(os.path.join(CACHE, "p_lgb_ho.npy"), p_lgb_ho)
    np.save(os.path.join(CACHE, "p_nn_ho.npy"),
            p_nn_ho if p_nn_ho is not None else np.zeros((1, 1)))
    np.save(os.path.join(CACHE, "yho.npy"), yho)
    np.save(os.path.join(CACHE, "sho.npy"), sho)
    np.save(os.path.join(CACHE, "na.npy"), na)
    np.save(os.path.join(CACHE, "Xho.npy"), Xho.values.astype(np.float32))
    import pickle
    with open(os.path.join(CACHE, "runs_ho.pkl"), "wb") as f:
        pickle.dump(runs_ho, f)
    np.save(os.path.join(CACHE, "ft_true.npy"), ho["fault_turn"].values)

    # ---------- отчёт с дефолтными настройками ----------
    import scipy.special as sp
    nn_w = float(meta.get("nn_weight", 0.7))
    offsets = np.asarray(meta.get("offsets_blend",
                                  meta.get("offsets_lgb", [0] * 7)), dtype=float)
    if p_nn_ho is not None:
        logits = nn_w * sp.logit(np.clip(p_nn_ho, 1e-6, 1 - 1e-6)) + \
                 (1 - nn_w) * sp.logit(np.clip(p_lgb_ho, 1e-6, 1 - 1e-6))
    else:
        logits = sp.logit(np.clip(p_lgb_ho, 1e-6, 1 - 1e-6))
    pred_idx = (logits + offsets).argmax(1)
    p_suc = build_success_probs(Xho.values, pred_idx, wdir)
    ps = (p_suc > 0.5).astype(int)
    ft_pred = localize_all(runs_ho, pred_idx)
    r = full_report(yho, pred_idx, sho, ps, na, ho["fault_turn"].values, ft_pred)
    print_report(r, "HOLDOUT seed=%d%s" % (args.seed, " (fast, LGBM only)" if args.no_nn else ""))

    per = Counter()
    for yi, pi in zip(yho, pred_idx):
        per[(LABELS[yi], LABELS[pi])] += 1
    conf = {k: v for k, v in per.items() if k[0] != k[1] and v >= 10}
    if conf:
        print("\nГлавные путаницы (true->pred >=10 раз):")
        for (a, b), v in sorted(conf.items(), key=lambda kv: -kv[1])[:8]:
            print("  %-16s -> %-16s %d" % (a, b, v))

    row = pd.DataFrame([{"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "cmd": "score", "seed": args.seed, **r}])
    write_history(row)
    return r


def write_history(row):
    if os.path.exists(HISTORY):
        row.to_csv(HISTORY, mode="a", header=False, index=False)
    else:
        row.to_csv(HISTORY, index=False)


def best_history():
    if os.path.exists(HISTORY):
        h = pd.read_csv(HISTORY)
        if len(h) and "score" in h:
            return float(h.score.max())
    return None


# ----------------------------------------------------------------- tune
def cmd_tune(args):
    import scipy.special as sp
    wdir = ensure_weights()
    meta = load_meta(wdir)
    p_lgb = np.load(os.path.join(CACHE, "p_lgb_ho.npy"))
    pnn_f = os.path.join(CACHE, "p_nn_ho.npy")
    p_nn = np.load(pnn_f)
    has_nn = p_nn.ndim == 2 and p_nn.shape[1] == 7
    yho = np.load(os.path.join(CACHE, "yho.npy"))
    sho = np.load(os.path.join(CACHE, "sho.npy"))
    na = np.load(os.path.join(CACHE, "na.npy"))
    Xho = np.load(os.path.join(CACHE, "Xho.npy"))
    ft_true = np.load(os.path.join(CACHE, "ft_true.npy"))
    import pickle
    with open(os.path.join(CACHE, "runs_ho.pkl"), "rb") as f:
        runs = pickle.load(f)
    mask = na >= 7

    lg_lgb = sp.logit(np.clip(p_lgb, 1e-6, 1 - 1e-6))
    lg_nn = sp.logit(np.clip(p_nn, 1e-6, 1 - 1e-6)) if has_nn else None

    loc_cache = {}

    def eval_knob(nn_w, offsets, th):
        logits = nn_w * lg_nn + (1 - nn_w) * lg_lgb if has_nn else lg_lgb
        pred = (logits + offsets).argmax(1)
        key = pred.tobytes()
        if key not in loc_cache:
            loc_cache[key] = localize_all(runs, pred)
        ft = loc_cache[key]
        p_suc = build_success_probs(Xho, pred, wdir)
        ps = (p_suc > th).astype(int)
        return full_report(yho, pred, sho, ps, na, ft_true, ft), pred

    # старт: текущие настройки
    nn_w0 = float(meta.get("nn_weight", 0.7))
    off0 = np.asarray(meta.get("offsets_blend",
                               meta.get("offsets_lgb", [0] * 7)), dtype=float)
    best_r, _ = eval_knob(nn_w0, off0, 0.5)
    best = {"nn_weight": nn_w0, "offsets": off0.tolist(), "th": 0.5, "rep": best_r}
    base_score = best_r["score"]
    print_report(best_r, "ДО тюнинга (текущие настройки)")

    # 1) вес NN
    if has_nn:
        for w in np.arange(0.50, 0.91, 0.05):
            r, _ = eval_knob(float(w), off0, 0.5)
            if r["score"] > best["rep"]["score"]:
                best.update(nn_weight=float(w), rep=r)
    # 2) порог success
    for th in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        r, _ = eval_knob(best["nn_weight"], np.asarray(best["offsets"]), th)
        if r["score"] > best["rep"]["score"]:
            best.update(th=th, rep=r)
    # 3) координатный подъём смещений (цель — композит)
    off = np.asarray(best["offsets"], dtype=float)
    for _ in range(3):
        improved = False
        for c in range(7):
            for d in (-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4):
                t = off.copy()
                t[c] += d
                r, _ = eval_knob(best["nn_weight"], t, best["th"])
                if r["score"] > best["rep"]["score"] + 1e-6:
                    best.update(offsets=t.tolist(), rep=r)
                    off = t
                    improved = True
        if not improved:
            break

    print_report(best["rep"], "ПОСЛЕ тюнинга")
    print("\nЛучшие настройки: nn_weight=%.2f  success_th=%.2f  offsets=%s"
          % (best["nn_weight"], best["th"], np.round(best["offsets"], 2).tolist()))

    # печём в weights/ только если лучше
    prev_best = best_history()
    accept = best["rep"]["score"] >= base_score - 1e-9
    if accept:
        meta.update({"nn_weight": best["nn_weight"],
                     "offsets_blend": best["offsets"],
                     "success_threshold": best["th"],
                     "tuned_on": "holdout seed %s" % args.seed,
                     "tuned_score": best["rep"]["score"]})
        save_meta(wdir, meta)
        os.makedirs(CACHE, exist_ok=True)
        with open(os.path.join(CACHE, "tuned_meta.json"), "w") as f:
            json.dump({"nn_weight": best["nn_weight"],
                       "offsets_blend": best["offsets"],
                       "success_threshold": best["th"]}, f)
        log("weights/meta.json обновлён")
    else:
        log("улучшения нет — настройки не менялись")
    row = pd.DataFrame([{"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "cmd": "tune", "seed": args.seed, **best["rep"]}])
    write_history(row)
    if prev_best is not None:
        print("История: прошлый лучший %.4f -> сейчас %.4f" % (prev_best, best["rep"]["score"]))


# ----------------------------------------------------------------- final
def cmd_final(args):
    wdir = ensure_weights()
    meta = load_meta(wdir)
    if args.retrain:
        subprocess.check_call([sys.executable, os.path.join(HERE, "pretrain.py"),
                               "--train", os.path.join(HERE, "data", "train.csv"),
                               "--out", wdir, "--folds", "5",
                               "--nn-epochs", "10", "--nn-seeds", "1,2,3",
                               "--jobs", str(args.jobs)])
        # pretrain перезапишет meta — вернуть тюнинговые ручки
        meta = load_meta(wdir)
        tuned = os.path.join(CACHE, "tuned_meta.json")
        if os.path.exists(tuned):
            with open(tuned) as f:
                meta.update(json.load(f))
            save_meta(wdir, meta)
    th = float(meta.get("success_threshold", 0.5))
    log("итоговые ручки: nn_weight=%.2f th=%.2f offsets=%s"
        % (float(meta.get("nn_weight", 0.7)), th,
           np.round(meta.get("offsets_blend", [0] * 7), 2).tolist()))
    # порог success зашиваем прямо в solution.py? Нет: solution использует 0.5.
    # Если тюнинг нашёл другой порог — пишем его в meta, solution.py его читает.
    out = os.path.join(HERE, "my_submission.zip")
    subprocess.check_call([sys.executable, os.path.join(HERE, "make_submission.py"),
                           "--weights", os.path.relpath(wdir, HERE), "--out", out])
    print("\nГотово: %s — загрузи его на платформу." % out)


# solution.py:读取 success threshold from meta (compat patch)
def patch_solution_threshold():
    """solution.py использует 0.5; делаем порог читаемым из meta."""
    p = os.path.join(HERE, "solution.py")
    src = open(p).read()
    if "success_threshold" in src:
        return
    src = src.replace(
        "    pred_success = (p_suc > 0.5).astype(int)",
        "    _th = float((meta or {}).get('success_threshold', 0.5))\n"
        "    pred_success = (p_suc > _th).astype(int)")
    open(p, "w").write(src)
    log("solution.py: порог success теперь берётся из meta.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score")
    sc.add_argument("--seed", type=int, default=42)
    sc.add_argument("--holdout", type=float, default=0.15)
    sc.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    sc.add_argument("--gpu", action="store_true")
    sc.add_argument("--no-nn", action="store_true")

    tu = sub.add_parser("tune")
    tu.add_argument("--seed", type=int, default=42)

    fi = sub.add_parser("final")
    fi.add_argument("--retrain", action="store_true")
    fi.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))

    args = ap.parse_args()
    patch_solution_threshold()
    if args.cmd == "score":
        cmd_score(args)
    elif args.cmd == "tune":
        cmd_tune(args)
    elif args.cmd == "final":
        cmd_final(args)


if __name__ == "__main__":
    main()
