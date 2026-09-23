#!/usr/bin/env python3
"""Diagnostic: is the mass cliff a norm/distribution mismatch, or missing info?

Follow-up to the E0 KILL and the layer-16 probe's success (see
docs/RESULTS.md). The probe proved layer 16's hidden state is linearly
decodable for gripper/phase state -- so the semantic information is there.
But lm_head + the frozen final norm still can't read it. This narrows down
*why*, between two candidates:

  (a) distribution mismatch: the frozen norm's learned per-dimension weight
      was calibrated for layer 32's activation statistics. Feed it a
      layer-16 vector with different per-dimension mean/scale and the
      output lands somewhere lm_head was never fit to interpret, even
      though the useful direction is still linearly present. Fixable, in
      principle, by an affine correction -- this is exactly the problem
      "tuned lens" (vs. naive "logit lens") was built to solve.
  (b) missing information: whatever subspace lm_head's fixed weight matrix
      actually reads to select a vocabulary token simply hasn't been
      populated yet at layer 16, even though OTHER decodable directions
      exist there (which is what the probe found -- it's free to use any
      discriminating direction, not constrained to lm_head's specific one).

This does not distinguish them by argument -- it tests (a) directly. Collect
depth-16 and depth-32 hidden vectors (the same one lm_head is fed, via the
same forward-hook approach as experiments/probe_layer16.py) on the same
items, compute each dimension's mean/std at both depths, and apply a
moment-matching affine correction to the depth-16 vectors before running
them through the real frozen norm + lm_head. If in-option mass and top-token
coherence recover substantially, that's (a). If they don't, whatever lm_head
needs isn't a scale/offset away from present -- more consistent with (b).

This is a crude stand-in for a real tuned lens (which fits the correction by
gradient descent against next-token loss, not just first and second
moments), so a negative result here does not rule out (a) as thoroughly as a
real tuned lens would -- it rules out the simplest version of it.

Usage
-----
    python experiments/diagnose_norm_mismatch.py \
        --manifests data/gripper.jsonl data/phase.jsonl \
        --out results/norm_mismatch_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from jev.calibration import softmax  # noqa: E402
from jev.evaluation import load_manifest, run_provenance  # noqa: E402
from jev.readout import OptionReadout  # noqa: E402


def collect_hidden_and_logits(readout: OptionReadout, items: list[dict]):
    """Hidden vector fed to lm_head, and lm_head's actual option logits, per item."""
    import torch
    from PIL import Image

    captured: list[torch.Tensor] = []

    def hook(_module, args):
        captured.append(args[0].detach())

    handle = readout.model.lm_head.register_forward_pre_hook(hook)
    try:
        device = next(readout.model.parameters()).device
        hidden_rows, option_logit_rows, mass_rows = [], [], []
        for rec in items:
            captured.clear()
            with Image.open(rec["image_path"]) as im:
                res = readout.score(im.convert("RGB"), rec["question"], rec["options"])
            hidden_rows.append(captured[0][0, -1, :].float().cpu().numpy())
            option_logit_rows.append(res.logits)
            mass_rows.append(res.meta["in_option_mass"])
        return np.stack(hidden_rows), np.stack(option_logit_rows), np.array(mass_rows)
    finally:
        handle.remove()


def apply_head(readout: OptionReadout, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the real frozen norm + lm_head on externally supplied hidden vectors.

    Bypasses the transformer stack entirely -- this is what makes it a clean
    test of the norm/lm_head specifically, decoupled from how the vector was
    produced or corrected.
    """
    import torch

    device = next(readout.model.parameters()).device
    x = torch.tensor(hidden, dtype=torch.float32, device=device)
    with torch.no_grad():
        normed = readout.model.model.text_model.norm(x)
        logits = readout.model.lm_head(normed)
    probs_full = torch.softmax(logits.float(), dim=-1)
    option_logits = logits[:, readout.option_token_ids].float().cpu().numpy()
    mass = probs_full[:, readout.option_token_ids].sum(dim=-1).cpu().numpy()
    top_ids = logits.argmax(dim=-1).cpu().tolist()
    top_tokens = [readout.tokenizer.decode([t]) for t in top_ids]
    return option_logits, mass, top_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/norm_mismatch_diagnostic.json"))
    args = ap.parse_args()

    # Manifests differ in option count (gripper: 2, phase: 3), so mass/top-token
    # evaluation (which depends on marker token ids) has to stay per-manifest,
    # with its own correctly-marked readout. Only the raw hidden-state arrays
    # get pooled, to fit one shared, more robust moment-matching correction.
    per_manifest = []  # (path, hidden_16, hidden_32, readout_for_scoring)
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

    hidden_16 = np.concatenate([p[1] for p in per_manifest], axis=0)
    hidden_32 = np.concatenate([p[2] for p in per_manifest], axis=0)
    n_items_total = sum(len(p[1]) for p in per_manifest)
    print(f"diagnostic: {n_items_total} items pooled across {len(args.manifests)} manifest(s) "
          f"for the moment-matching statistics")

    mean_16, std_16 = hidden_16.mean(0), hidden_16.std(0) + 1e-8
    mean_32, std_32 = hidden_32.mean(0), hidden_32.std(0) + 1e-8

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

    results = {}
    for path, h16, h32, readout in per_manifest:
        print(f"\n  {path}:")
        corrected_16 = (h16 - mean_16) / std_16 * std_32 + mean_32
        _, mass_raw, top_raw = apply_head(readout, h16)              # baseline: known FORMAT_FAILURE
        _, mass_corrected, top_corrected = apply_head(readout, corrected_16)  # the test
        _, mass_ref, top_ref = apply_head(readout, h32)              # ceiling: known-coherent reference

        row_raw = summarize(mass_raw, top_raw, "raw depth-16 (baseline)")
        row_corrected = summarize(mass_corrected, top_corrected, "moment-matched depth-16 (the test)")
        row_ref = summarize(mass_ref, top_ref, "true depth-32 (ceiling)")

        verdict = (
            "DISTRIBUTION_MISMATCH"
            if row_corrected["mean_mass"] > 10 * max(row_raw["mean_mass"], 1e-6)
            and row_corrected["mean_mass"] > 0.05
            else "NOT_EXPLAINED_BY_MOMENTS"
        )
        print(f"    verdict: {verdict}")
        results[path] = {
            "n_items": len(h16),
            "raw_depth16": row_raw,
            "moment_matched_depth16": row_corrected,
            "true_depth32": row_ref,
            "verdict": verdict,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "manifests": [str(m) for m in args.manifests],
                "n_items_total": n_items_total,
                "depth_tested": args.depth,
                "provenance": run_provenance({"model_id": args.model_id, "device": args.device}),
                "per_manifest": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
