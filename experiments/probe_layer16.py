#!/usr/bin/env python3
"""Supplementary check: is layer-16's hidden state linearly decodable at all?

Not part of the pre-registered E0/E1/E2 line -- this is a follow-up to a
KILL verdict on E0 and E1 (see docs/RESULTS.md), asked because the KILL
result is ambiguous about *why* the frozen lm_head fails. Two very different
explanations are consistent with "lm_head decodes noise from layer 16":

  (a) layer 16's hidden state carries the semantic signal (gripper state,
      task phase) just fine, but lm_head -- frozen at pretraining, never
      adapted to a truncated stream -- has no idea how to read it.
  (b) layer 16 does not carry that signal in an accessible form at all, and
      the action expert is grounding on something else entirely (low-level
      visual/proprioceptive cues, not semantic content).

(a) is what the whole readout premise implicitly assumed when it noted the
policy "works" as indirect evidence the representation must be informative.
This script tests that assumption directly, on the exact hidden vector
lm_head was fed (captured via a forward hook on lm_head's input, not
recomputed independently) instead of inferring it from lm_head's output.

This is deliberately NOT presented as a JEV-style free readout. A trained
linear probe on hidden states is SAFE/VLAConf territory -- a different,
already-published category of contribution -- named as such here per
docs/CONTEXT.md section 6, not smuggled in as a revival of the zero-training
premise.

Usage
-----
    python experiments/probe_layer16.py \
        --manifest data/gripper.jsonl --depths 16 32 \
        --out results/probe_gripper.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from jev.evaluation import load_manifest, manifest_fingerprint, run_provenance  # noqa: E402
from jev.readout import OptionReadout  # noqa: E402


def collect_hidden_states(readout: OptionReadout, items: list[dict]) -> np.ndarray:
    """Hook lm_head's input to capture the exact vector it was fed, per item.

    Reading lm_head's input directly (rather than recomputing "the layer-16
    hidden state" independently) guarantees this is testing the same vector
    the readout's KILL verdict was measured on -- no risk of a norm-placement
    or indexing mismatch silently comparing two different things.
    """
    import torch
    from PIL import Image

    captured: list[torch.Tensor] = []

    def hook(_module, args, _kwargs):
        captured.append(args[0].detach())

    handle = readout.model.lm_head.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        device = next(readout.model.parameters()).device
        rows = []
        for rec in items:
            captured.clear()
            with Image.open(rec["image_path"]) as im:
                inputs = readout.build_prompt(rec["question"], rec["options"], im.convert("RGB"))
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                readout.model(**inputs, use_cache=False)
            # same decision position collect_logits reads: the last prompt token
            rows.append(captured[0][0, -1, :].float().cpu().numpy())
        return np.stack(rows)
    finally:
        handle.remove()


def fit_probe(features: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    """L2-logistic probe, split-half like E0/E1: fit on one half, score the other.

    960 features on ~150 training items is badly underdetermined without
    regularization, hence LogisticRegressionCV picking C by internal
    cross-validation on the train half rather than a fixed guess.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    n = len(labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = n // 2
    train, test = perm[:cut], perm[cut:]

    scaler = StandardScaler().fit(features[train])
    x_train, x_test = scaler.transform(features[train]), scaler.transform(features[test])

    clf = LogisticRegressionCV(
        Cs=np.logspace(-4, 2, 13),
        cv=5,
        max_iter=5000,
        random_state=seed,
    ).fit(x_train, labels[train])

    pred = clf.predict(x_test)
    acc = float((pred == labels[test]).mean())
    proba = clf.predict_proba(x_test)
    k = proba.shape[1]
    try:
        if k == 2:
            auroc = float(roc_auc_score(labels[test], proba[:, 1]))
        else:
            auroc = float(roc_auc_score(labels[test], proba, multi_class="ovr"))
    except ValueError:
        auroc = None  # a class missing from the test half; small-n hazard, not a bug

    return {
        "n_total": int(n),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "k": int(k),
        "chance": 1.0 / k,
        "accuracy": acc,
        "auroc": auroc,
        "best_C": float(clf.C_[0]) if hasattr(clf, "C_") else None,
    }


def permutation_null(features: np.ndarray, labels: np.ndarray, seed: int, n_perms: int = 20) -> dict:
    """Refit the same pipeline under label permutation: what does pure overfitting buy?

    960 features on ~150 training items is a regime where a regularized linear
    model can look like it's found signal from noise alone. Holding the
    train/test split fixed and only permuting labels isolates that risk: if
    the real accuracy sits inside the null distribution the permutations
    produce, the probe result is not distinguishable from overfitting, no
    matter how it looks in isolation.
    """
    rng = np.random.default_rng(seed + 1)
    null_accs = []
    for i in range(n_perms):
        shuffled = labels[rng.permutation(len(labels))]
        row = fit_probe(features, shuffled, seed=seed)
        null_accs.append(row["accuracy"])
    return {"n_perms": n_perms, "null_accuracies": null_accs,
            "null_mean": float(np.mean(null_accs)), "null_std": float(np.std(null_accs))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    ap.add_argument("--depths", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/probe.json"))
    args = ap.parse_args()

    items = load_manifest(args.manifest)
    markers = tuple("ABCDEFGH"[: len(items[0]["options"])])
    labels = np.array([r["label"] for r in items])
    print(f"probe: {len(items)} items, {len(markers)} options, depths={args.depths}")

    results = {}
    for depth in sorted(set(args.depths)):
        print(f"  depth {depth:>2} ... ", end="", flush=True)
        readout = OptionReadout.from_pretrained(
            args.model_id, truncate_layers=depth, device=args.device, dtype=args.dtype,
            markers=markers, leading_space=True,
        )
        features = collect_hidden_states(readout, items)
        row = fit_probe(features, labels, seed=args.seed)
        row["null"] = permutation_null(features, labels, seed=args.seed)
        results[depth] = row
        auroc_str = f"{row['auroc']:.3f}" if row["auroc"] is not None else "n/a"
        z = (row["accuracy"] - row["null"]["null_mean"]) / max(row["null"]["null_std"], 1e-9)
        print(
            f"acc={row['accuracy']:.3f} (chance={row['chance']:.3f}) auroc={auroc_str} "
            f"C={row['best_C']:.4g} | null_acc={row['null']['null_mean']:.3f}"
            f"+/-{row['null']['null_std']:.3f} z={z:.2f}"
        )
        del readout

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "n_items": len(items),
                "manifest_fingerprint": manifest_fingerprint(items),
                "provenance": run_provenance(
                    {"model_id": args.model_id, "device": args.device, "dtype": args.dtype,
                     "note": "linear probe on lm_head's input activations, not part of E0/E1/E2"}
                ),
                "depths": {str(d): r for d, r in results.items()},
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
