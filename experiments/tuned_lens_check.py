#!/usr/bin/env python3
"""Full affine (tuned-lens-style) correction vs. the diagonal one, held out fairly.

experiments/diagnose_norm_mismatch.py showed a *diagonal* correction (rescale
each of 960 dimensions independently by its own mean/std) recovers most of
the frozen head's format quality. That's evidence the depth-16 and depth-32
representations share the same coordinate axes, just at different scales --
because a correction that can't mix dimensions together shouldn't be able to
fix a representation whose information is genuinely *reorganized* across
axes between the two depths. If it were reorganized, a diagonal fix should
fail where a full linear map succeeds.

This tests that directly: fit a full, regularized affine map
`y = Wx + b` (ordinary tuned lens is exactly this, normally fit by gradient
descent against next-token loss; ridge regression against the target
activations themselves is a reasonable, cheaper stand-in) from depth-16
vectors to depth-32 vectors, and compare its recovered mass/coherence
against the diagonal correction -- both evaluated on a held-out test split
never seen during fitting, since a full 960x960 map has ~922k parameters
against a few hundred training items and will overfit given the chance.

If the full map does meaningfully better than diagonal-only: real evidence
for a rotational/reorganization component. If it does about the same:
confirms the diagonal story -- same axes, different scale, nothing more
exotic going on.

Usage
-----
    python experiments/tuned_lens_check.py \
        --manifests data/gripper.jsonl data/phase.jsonl \
        --out results/tuned_lens_check.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from diagnose_norm_mismatch import apply_head, collect_hidden_and_logits  # noqa: E402
from jev.evaluation import load_manifest, run_provenance  # noqa: E402
from jev.readout import OptionReadout  # noqa: E402


def summarize(mass, tops, label):
    from collections import Counter

    top_counts = Counter(tops)
    print(f"    [{label}] mean_mass={mass.mean():.4f} median_mass={np.median(mass):.4f} "
          f"top5_tokens={top_counts.most_common(5)}")
    return {
        "mean_mass": float(mass.mean()),
        "median_mass": float(np.median(mass)),
        "frac_above_0.05": float((mass > 0.05).mean()),
        "top_token_counts": dict(top_counts.most_common(10)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/tuned_lens_check.json"))
    args = ap.parse_args()

    per_manifest = []  # (path, hidden_16, hidden_32, readout_for_scoring, n_items)
    for manifest_path in args.manifests:
        m_items = load_manifest(manifest_path)
        markers = tuple("ABCDEFGH"[: len(m_items[0]["options"])])
        print(f"  {manifest_path}: {len(m_items)} items, markers={markers}")

        truncated = OptionReadout.from_pretrained(
            args.model_id, truncate_layers=args.depth, device=args.device, dtype="float32",
            markers=markers, leading_space=True,
        )
        h16, _, _ = collect_hidden_and_logits(truncated, m_items)
        del truncated

        full_m = OptionReadout.from_pretrained(
            args.model_id, truncate_layers=32, device=args.device, dtype="float32",
            markers=markers, leading_space=True,
        )
        h32, _, _ = collect_hidden_and_logits(full_m, m_items)
        per_manifest.append((str(manifest_path), h16, h32, full_m))

    # Pool for fitting (the map is a property of the model's geometry, not the
    # task), split by a fixed seed so train/test membership is consistent
    # regardless of which manifest an item came from.
    hidden_16 = np.concatenate([p[1] for p in per_manifest], axis=0)
    hidden_32 = np.concatenate([p[2] for p in per_manifest], axis=0)
    boundaries = np.cumsum([0] + [len(p[1]) for p in per_manifest])
    n_total = len(hidden_16)
    print(f"\nfitting: {n_total} items pooled across {len(args.manifests)} manifest(s)")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    cut = n_total // 2
    train_idx, test_idx = set(perm[:cut].tolist()), set(perm[cut:].tolist())

    # Diagonal correction, fit on train only (matches the ridge fit's held-out
    # discipline, even though 1920 parameters vs ~300 items has little
    # overfitting risk on its own -- keeping both corrections to the same
    # standard makes the comparison clean).
    train_mask = np.array([i in train_idx for i in range(n_total)])
    mean_16, std_16 = hidden_16[train_mask].mean(0), hidden_16[train_mask].std(0) + 1e-8
    mean_32, std_32 = hidden_32[train_mask].mean(0), hidden_32[train_mask].std(0) + 1e-8

    # Full affine (tuned-lens-style) correction, ridge-regularized, CV-selected alpha.
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    x_scaler = StandardScaler().fit(hidden_16[train_mask])
    y_scaler = StandardScaler().fit(hidden_32[train_mask])
    x_train = x_scaler.transform(hidden_16[train_mask])
    y_train = y_scaler.transform(hidden_32[train_mask])

    print("  fitting ridge tuned lens (960 -> 960, CV over alpha) ...")
    lens = RidgeCV(alphas=np.logspace(1, 6, 12)).fit(x_train, y_train)
    print(f"  selected alpha={lens.alpha_:.4g}, train R^2={lens.score(x_train, y_train):.4f}")

    results = {}
    offset = 0
    for path, h16, h32, readout in per_manifest:
        n = len(h16)
        idx_here = np.arange(offset, offset + n)
        test_here = np.array([i in test_idx for i in idx_here])
        offset += n
        print(f"\n  {path} ({test_here.sum()} held-out test items):")

        h16_test, h32_test = h16[test_here], h32[test_here]

        diag_corrected = (h16_test - mean_16) / std_16 * std_32 + mean_32
        lens_corrected = y_scaler.inverse_transform(lens.predict(x_scaler.transform(h16_test)))

        _, mass_raw, top_raw = apply_head(readout, h16_test)
        _, mass_diag, top_diag = apply_head(readout, diag_corrected)
        _, mass_lens, top_lens = apply_head(readout, lens_corrected)
        _, mass_ref, top_ref = apply_head(readout, h32_test)

        row_raw = summarize(mass_raw, top_raw, "raw depth-16")
        row_diag = summarize(mass_diag, top_diag, "diagonal correction")
        row_lens = summarize(mass_lens, top_lens, "ridge tuned lens")
        row_ref = summarize(mass_ref, top_ref, "true depth-32")

        gain_diag = row_diag["mean_mass"] - row_raw["mean_mass"]
        gain_lens_over_diag = row_lens["mean_mass"] - row_diag["mean_mass"]
        print(f"    diagonal gain over raw: {gain_diag:+.4f} | "
              f"tuned-lens gain OVER diagonal: {gain_lens_over_diag:+.4f}")

        results[path] = {
            "n_test": int(test_here.sum()),
            "raw_depth16": row_raw,
            "diagonal_correction": row_diag,
            "ridge_tuned_lens": row_lens,
            "true_depth32": row_ref,
            "diagonal_gain_over_raw": gain_diag,
            "tuned_lens_gain_over_diagonal": gain_lens_over_diag,
        }

    verdict = (
        "ROTATIONAL_COMPONENT_FOUND"
        if any(r["tuned_lens_gain_over_diagonal"] > 0.05 for r in results.values())
        else "DIAGONAL_EXPLAINS_IT"
    )
    print(f"\noverall verdict: {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "manifests": [str(m) for m in args.manifests],
                "n_items_total": n_total,
                "depth_tested": args.depth,
                "ridge_alpha": float(lens.alpha_),
                "ridge_train_r2": float(lens.score(x_train, y_train)),
                "provenance": run_provenance({"model_id": args.model_id, "device": args.device}),
                "per_manifest": results,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
