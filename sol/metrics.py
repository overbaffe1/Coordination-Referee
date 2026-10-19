"""Metric definitions for the Coordination Referee competition."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]

# composite leaderboard score
WEIGHTS = {"macro_f1": 0.50, "robustness_f1": 0.25, "success_f1": 0.15,
           "fault_turn_hit2": 0.10}


def f1_per_class(y_true: Sequence[str], y_pred: Sequence[str], labels) -> Dict[str, float]:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    out = {}
    for c in labels:
        denom = 2 * tp[c] + fp[c] + fn[c]
        out[c] = (2.0 * tp[c] / denom) if denom else 0.0
    return out


def macro_f1(y_true, y_pred, labels=None) -> float:
    labels = labels or LABELS
    per = f1_per_class(y_true, y_pred, labels)
    present = [c for c in labels if any(t == c for t in y_true)] or labels
    return sum(per[c] for c in present) / len(present)


def binary_f1(y_true: Sequence[int], y_pred: Sequence[int], positive: int = 1) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom else 0.0


def fault_turn_hit_at_k(y_true_label, y_pred_label, t_true, t_pred, k: int = 2) -> float:
    """Share of *truly faulty* runs where the predicted turn is within +/-k.

    A run whose class was misclassified counts as a miss, so localisation can
    never be farmed independently of classification.
    """
    hits = total = 0
    for lt, lp, tt, tp in zip(y_true_label, y_pred_label, t_true, t_pred):
        if lt == "clean":
            continue
        total += 1
        if lp == lt and tp is not None and int(tp) >= 0 and abs(int(tp) - int(tt)) <= k:
            hits += 1
    return hits / total if total else 0.0


def group_f1(y_true, y_pred, groups) -> Dict[str, float]:
    """Macro F1 inside every group (used for the robustness diagnostics)."""
    buckets = defaultdict(lambda: ([], []))
    for t, p, g in zip(y_true, y_pred, groups):
        buckets[g][0].append(t)
        buckets[g][1].append(p)
    return {g: macro_f1(t, p) for g, (t, p) in buckets.items()}


def composite(scores: Dict[str, float]) -> float:
    return sum(w * scores.get(k, 0.0) for k, w in WEIGHTS.items())
