#!/usr/bin/env python3
"""E0 -- the kill test: does a SmolVLA-truncated backbone still read out?

Why this is the first experiment and not the last
-------------------------------------------------
The whole "JEV readout on a VLA" premise assumes you can ask the policy's own
backbone a semantic question in one extra forward pass. For SmolVLA you
cannot, naively: LeRobot builds SmolVLA by keeping only the first 16 of
SmolLM2-360M's 32 text layers (smolvlm_with_expert.py, `num_vlm_layers=16`).
The lm_head and final norm survive -- SmolVLA freezes both -- but they were
trained to decode a 32-layer residual stream and would now be fed a 16-layer
one.

So before any robot, any LIBERO rollout, any GPU-hours: measure whether
depth-truncated + original head still produces discriminative, calibratable
option-token distributions. This needs one 500M model and a few hundred
labelled MCQA items. It runs on CPU. If it fails, the architecture question is
settled and you pivot to a parallel full-depth VLM without having collected a
single rollout.

Pre-registered decision rule (fix this before you look at the numbers)
---------------------------------------------------------------------
Let k be the number of options and D=16 the SmolVLA depth.

  FORMAT FAILURE  in_option_mass(D) < 0.05
      The model is not answering in the requested format. This is a prompt
      bug, not a depth result. Fix the prompt and rerun; do not report.

  KILL           accuracy(D) <= chance + 2*se  OR  auroc_recal(D) <= 0.55
      The truncated stack carries no head-reachable semantic signal. The
      readout must not live on SmolVLA's backbone. Pivot to (a) a parallel
      full-depth VLM, or (b) a trained probe on layer-16 hidden states --
      note that (b) is SAFE/VLAConf territory and is no longer a JEV readout.

  SURVIVES       accuracy(D) well above chance AND auroc_recal(D) >= 0.65
      Proceed to E1 (does action-finetuning move it?) and E2 (does it gate?).

  INTERESTING    accuracy(D) above chance but ECE_raw(D) >> ECE_raw(32) and
                 recalibration closes most of the gap
      This is the publishable middle: truncation costs calibration, not
      content, and post-hoc scaling buys it back. Say so explicitly.

Usage
-----
    python experiments/e0_depth_ablation.py \
        --manifest data/mcqa.jsonl \
        --depths 8 12 16 20 24 32 \
        --out results/e0.json

Manifest format (one JSON object per line):
    {"image": "frames/0001.png",
     "question": "Is the gripper on track to grasp the red block?",
     "options": ["on track", "blocked", "unsafe"],
     "label": 0}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from jev.evaluation import (  # noqa: E402
    collect_logits,
    evaluate,
    load_manifest,
    manifest_fingerprint,
    run_provenance,
    verdict,
)
from jev.readout import OptionReadout  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    ap.add_argument("--depths", type=int, nargs="+", default=[8, 12, 16, 20, 24, 32])
    ap.add_argument("--smolvla-depth", type=int, default=16, help="depth under test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/e0.json"))
    args = ap.parse_args()

    items = load_manifest(args.manifest)
    markers = tuple("ABCDEFGH"[: len(items[0]["options"])])
    print(f"E0: {len(items)} items, {len(markers)} options, markers={markers}")

    # Determine leading_space once, at full depth, and apply it to every depth in
    # the sweep. Autodetecting per-iteration-but-only-on-the-last-depth (the
    # previous behaviour) scored every other depth with the wrong tokenization,
    # which silently misreports them as FORMAT_FAILURE instead of measuring
    # truncation. The tokenization convention should not depend on depth; only
    # whether the resulting answer is any good should.
    from PIL import Image

    probe = OptionReadout.from_pretrained(
        args.model_id,
        truncate_layers=max(args.depths),
        device=args.device,
        dtype=args.dtype,
        markers=markers,
    )
    with Image.open(items[0]["image_path"]) as probe_im:
        leading_space, probe_mass = probe.autodetect_leading_space(
            probe_im.convert("RGB"), items[0]["question"], items[0]["options"]
        )
    print(f"  [leading_space={leading_space}, probe_mass={probe_mass:.3f}]")
    del probe

    results = {}
    for depth in sorted(set(args.depths)):
        print(f"  depth {depth:>2} ... ", end="", flush=True)
        readout = OptionReadout.from_pretrained(
            args.model_id,
            truncate_layers=depth,
            device=args.device,
            dtype=args.dtype,
            markers=markers,
            leading_space=leading_space,
        )
        logits, labels, mass = collect_logits(readout, items)
        row = evaluate(logits, labels, seed=args.seed)
        row["in_option_mass"] = mass
        row["verdict"] = verdict(row, mass)
        results[depth] = row
        print(
            f"acc={row['accuracy']:.3f} ece={row['ece_raw']:.3f}->{row['ece_recal']:.3f} "
            f"auroc={row['auroc_recal']:.3f} T={row['temperature']:.2f} [{row['verdict']}]"
        )
        del readout

    target = results.get(args.smolvla_depth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "manifest": str(args.manifest),
                "n_items": len(items),
                "manifest_fingerprint": manifest_fingerprint(items),
                "provenance": run_provenance(
                    {"model_id": args.model_id, "device": args.device, "dtype": args.dtype}
                ),
                "smolvla_depth": args.smolvla_depth,
                "depths": {str(d): r for d, r in results.items()},
                "headline_verdict": target["verdict"] if target else "NOT_RUN",
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")
    if target:
        print(f"VERDICT at SmolVLA depth {args.smolvla_depth}: {target['verdict']}")
        return 0 if target["verdict"] in {"SURVIVES", "INCONCLUSIVE"} else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
