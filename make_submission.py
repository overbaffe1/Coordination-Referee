#!/usr/bin/env python3
"""Build the submission zip for the platform.

    python3 make_submission.py --weights weights_demo --out submission.zip

Zip layout (flat, solution.py at the root):
    solution.py          inference entry point (contract of the platform)
    engine.py            message -> event parser
    features4.py         structural features
    nn_model.py          BiGRU sequence model
    localize.py          fault_turn rules
    pretrain.py          offline training script (reproducibility, p.6 of rules)
    weights/             pre-trained models + meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ["engine.py", "features4.py", "nn_model.py", "localize.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", default="submission.zip")
    args = ap.parse_args()

    wdir = os.path.join(HERE, args.weights)
    meta_path = os.path.join(wdir, "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit("no meta.json in %s — run pretrain.py first" % wdir)
    with open(meta_path) as f:
        meta = json.load(f)
    print("weights: OOF blend %.4f | lgb %.4f | nn seeds %s" %
          (meta.get("oof_macro_f1_blend", -1), meta.get("oof_macro_f1_lgb", -1),
           meta.get("nn_seeds", [])))

    items = [os.path.join(HERE, "solution.py"), os.path.join(HERE, "pretrain.py")]
    items += [os.path.join(HERE, "sol", m) for m in MODULES]
    for root, _, files in os.walk(wdir):
        for fn in files:
            if fn.startswith("_oof_"):
                continue          # debug artifacts, not needed for inference
            items.append(os.path.join(root, fn))

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in items:
            arc = os.path.relpath(path, HERE)
            z.write(path, arc)
            print("  +", arc)
    size = os.path.getsize(args.out) / 1e6
    print("wrote %s (%.1f MB)" % (args.out, size))


if __name__ == "__main__":
    main()
