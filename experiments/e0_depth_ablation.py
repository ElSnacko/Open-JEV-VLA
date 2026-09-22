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
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from jev.calibration import (  # noqa: E402
    auroc,
    brier_score,
    calibrate_gate,
    expected_calibration_error,
    nll,
    softmax,
)
from jev.readout import OptionReadout  # noqa: E402

CHANCE_MARGIN_SE = 2.0
AUROC_KILL = 0.55
AUROC_PASS = 0.65
FORMAT_FLOOR = 0.05


def load_manifest(path: pathlib.Path) -> list[dict]:
    root = path.parent
    items = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["image_path"] = str((root / rec["image"]).resolve())
            items.append(rec)
    if not items:
        raise SystemExit(f"no items in {path}")
    n_opts = {len(r["options"]) for r in items}
    if len(n_opts) != 1:
        raise SystemExit(f"all items must share option count, saw {sorted(n_opts)}")
    return items


def collect_logits(readout: OptionReadout, items: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    from PIL import Image

    rows, masses = [], []
    for rec in items:
        with Image.open(rec["image_path"]) as im:
            res = readout.score(im.convert("RGB"), rec["question"], rec["options"])
        rows.append(res.logits)
        masses.append(res.meta["in_option_mass"])
    labels = np.array([r["label"] for r in items])
    return np.stack(rows), labels, float(np.mean(masses))


def evaluate(logits: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict:
    """Split-half: fit recalibration on one half, report on the other."""
    n, k = logits.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = n // 2
    cal, test = perm[:cut], perm[cut:]

    raw = softmax(logits[test])
    correct_raw = raw.argmax(1) == labels[test]

    gate = calibrate_gate(logits[cal], labels[cal], alpha=0.1)
    recal = gate.probabilities(logits[test])
    correct_recal = recal.argmax(1) == labels[test]

    acc = float(correct_raw.mean())
    se = math.sqrt(max(acc * (1 - acc), 1e-12) / max(len(test), 1))

    act, set_size = gate.decide(logits[test])

    return {
        "n_total": int(n),
        "n_test": int(len(test)),
        "k": int(k),
        "chance": 1.0 / k,
        "accuracy": acc,
        "accuracy_se": se,
        "accuracy_recal": float(correct_recal.mean()),
        "ece_raw": expected_calibration_error(raw, labels[test]),
        "ece_recal": expected_calibration_error(recal, labels[test]),
        "brier_raw": brier_score(raw, labels[test]),
        "brier_recal": brier_score(recal, labels[test]),
        "nll_raw": nll(raw, labels[test]),
        "nll_recal": nll(recal, labels[test]),
        # does confidence rank correctness? this is the gating-relevant number
        "auroc_raw": auroc(raw.max(1), correct_raw),
        "auroc_recal": auroc(recal.max(1), correct_recal),
        "temperature": gate.temperature,
        "conformal_qhat": gate.qhat,
        "mean_set_size": float(set_size.mean()),
        "act_rate": float(act.mean()),
    }


def verdict(row: dict, in_option_mass: float) -> str:
    if in_option_mass < FORMAT_FLOOR:
        return "FORMAT_FAILURE"
    above_chance = row["accuracy"] > row["chance"] + CHANCE_MARGIN_SE * row["accuracy_se"]
    if not above_chance or row["auroc_recal"] <= AUROC_KILL:
        return "KILL"
    if row["auroc_recal"] >= AUROC_PASS:
        return "SURVIVES"
    return "INCONCLUSIVE"


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

    results = {}
    for depth in sorted(set(args.depths)):
        print(f"  depth {depth:>2} ... ", end="", flush=True)
        readout = OptionReadout.from_pretrained(
            args.model_id,
            truncate_layers=depth,
            device=args.device,
            dtype=args.dtype,
            markers=markers,
        )
        if depth == max(args.depths):
            from PIL import Image

            with Image.open(items[0]["image_path"]) as probe_im:
                ls, probe_mass = readout.autodetect_leading_space(
                    probe_im.convert("RGB"),
                    items[0]["question"],
                    items[0]["options"],
                )
            print(f"[leading_space={ls}, mass={probe_mass:.3f}] ", end="", flush=True)

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
