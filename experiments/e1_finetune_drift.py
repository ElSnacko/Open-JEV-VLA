#!/usr/bin/env python3
"""E1 -- does action finetuning move the readout, beyond what truncation cost?

E0 answers whether a half-depth stack can be read out at all, using the base
VLM's own weights. E1 asks the next question: SmolVLA's 16 layers are not the
base model's 16 layers any more. Community-scale action pretraining has moved
them. Did that break the readout, leave it intact, or (the interesting case)
leave it intact but miscalibrated in a way post-hoc scaling can fix?

Why this is not already answered
--------------------------------
The literature establishes that action finetuning damages VLM representations
in general: AEGIS (2604.16067) reports VQA holdout loss degrading steadily
within 1,500 steps of naive finetuning, and Knowledge Insulation (2505.23705)
exists specifically to stop the continuous action expert's gradient reaching
the VLM. What nobody has measured is the effect on *option-token calibration*
at a decision position, which is the quantity a JEV-style gate depends on. A
model can lose VQA loss and keep a usable ranking; it can also keep accuracy
and lose calibration. Those are different failures with different fixes.

One thing genuinely in our favour: SmolVLA freezes `lm_head`, `text_model.norm`
and the last two kept layers during action training
(smolvlm_with_expert.py:163-174). So the decoding head and the layers feeding
it are the original pretrained weights. Everything below them has moved. That
makes intact-readout a live possibility rather than wishful thinking.

Weight variants compared, all at the same depth, on the same items
------------------------------------------------------------------
    base      SmolVLM2-500M truncated to 16 layers. E0's reference point.
    smolvla   lerobot/smolvla_base. Community-dataset action pretraining.
    libero    lerobot/smolvla_libero. Further finetuned on LIBERO, so it
              carries strictly more action-training exposure than smolvla_base
              and gives the comparison a gradient rather than two points.

Reading the output
------------------
Per variant: accuracy, ECE before and after recalibration, AUROC of confidence
against correctness, fitted temperature. Plus, against `base`:

    top1_agreement   how often the two variants pick the same option
    mean_sym_kl      how far the distributions moved, regardless of quality

The four outcomes worth naming in advance:

    INTACT       accuracy and AUROC hold within noise of base. Readout can
                 live on the policy. Best case, and the paper writes itself.
    RECALIBRABLE accuracy holds, ECE degrades, recalibration restores it.
                 Also a good result: it means the gate needs a calibration
                 step per checkpoint, which is cheap and expected anyway.
    DEGRADED     accuracy drops toward chance, or AUROC falls below 0.55.
                 The readout cannot live on an action-finetuned policy. Report
                 it: this is the architecture decision the whole line of work
                 needs, and it redirects the field to a parallel VLM.
    MOVED-BUT-OK high symmetric KL with preserved AUROC. The distribution
                 shifted a lot but the ranking survived, which says calibrate
                 per checkpoint and never transfer a threshold across them.

Usage
-----
    python experiments/e1_finetune_drift.py \
        --manifest data/phase.jsonl \
        --variants base smolvla libero \
        --out results/e1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from jev.calibration import softmax  # noqa: E402
from jev.evaluation import (  # noqa: E402
    collect_logits,
    evaluate,
    load_manifest,
    manifest_fingerprint,
    run_provenance,
    symmetric_kl,
    verdict,
)
from jev.readout import OptionReadout  # noqa: E402

VARIANTS = {
    "base": {"kind": "vlm", "id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"},
    "smolvla": {"kind": "smolvla", "id": "lerobot/smolvla_base"},
    "libero": {"kind": "smolvla", "id": "lerobot/smolvla_libero"},
}

# how far AUROC may fall from base before we call it degraded
AUROC_DROP_TOLERANCE = 0.05
# symmetric KL above this counts as "the distribution really moved"
KL_MOVED = 0.5


def load_variant(
    name: str, depth: int, device: str, dtype: str, markers, leading_space: bool
) -> OptionReadout:
    spec = VARIANTS[name]
    if spec["kind"] == "vlm":
        return OptionReadout.from_pretrained(
            spec["id"],
            truncate_layers=depth,
            device=device,
            dtype=dtype,
            markers=markers,
            leading_space=leading_space,
        )
    # SmolVLA checkpoints arrive already truncated by their own config
    return OptionReadout.from_smolvla(
        spec["id"], device=device, markers=markers, leading_space=leading_space
    )


def classify(row: dict, base_row: dict, mean_kl: float) -> str:
    if row["verdict"] == "FORMAT_FAILURE":
        return "FORMAT_FAILURE"
    auroc_drop = base_row["auroc_recal"] - row["auroc_recal"]
    near_chance = row["accuracy"] <= row["chance"] + 2 * row["accuracy_se"]
    if near_chance or row["auroc_recal"] <= 0.55 or auroc_drop > 2 * AUROC_DROP_TOLERANCE:
        return "DEGRADED"
    if mean_kl > KL_MOVED:
        return "MOVED-BUT-OK"
    if row["ece_raw"] > base_row["ece_raw"] * 2 and row["ece_recal"] <= base_row["ece_recal"] * 1.5:
        return "RECALIBRABLE"
    if auroc_drop <= AUROC_DROP_TOLERANCE:
        return "INTACT"
    return "INCONCLUSIVE"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["base", "smolvla"], choices=list(VARIANTS))
    ap.add_argument("--depth", type=int, default=16, help="truncation depth for the base variant")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/e1.json"))
    args = ap.parse_args()

    if "base" not in args.variants:
        raise SystemExit("`base` is the reference point; include it in --variants")

    items = load_manifest(args.manifest)
    markers = tuple("ABCDEFGH"[: len(items[0]["options"])])
    print(f"E1: {len(items)} items, {len(markers)} options, depth {args.depth}")

    # Determine leading_space once, on the base variant, and hold it fixed across
    # every variant. Letting each variant pick its own tokenization convention
    # would make the cross-variant comparison meaningless -- a difference in
    # in_option_mass could then come from the markers, not from the checkpoint.
    from PIL import Image

    probe = load_variant("base", args.depth, args.device, args.dtype, markers, False)
    with Image.open(items[0]["image_path"]) as probe_im:
        leading_space, probe_mass = probe.autodetect_leading_space(
            probe_im.convert("RGB"), items[0]["question"], items[0]["options"]
        )
    print(f"  [leading_space={leading_space}, probe_mass={probe_mass:.3f}]")
    del probe

    probs_by_variant: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}

    for name in args.variants:
        print(f"  {name:<8} ... ", end="", flush=True)
        readout = load_variant(name, args.depth, args.device, args.dtype, markers, leading_space)
        n_layers = None
        try:
            from jev.readout import n_text_layers

            n_layers = n_text_layers(readout.model)
        except Exception:  # noqa: BLE001 - informational only
            pass

        logits, labels, mass = collect_logits(readout, items)
        row = evaluate(logits, labels, seed=args.seed)
        row["in_option_mass"] = mass
        row["n_text_layers"] = n_layers
        row["verdict"] = verdict(row, mass)
        results[name] = row
        probs_by_variant[name] = softmax(logits)
        print(
            f"layers={n_layers} acc={row['accuracy']:.3f} "
            f"ece={row['ece_raw']:.3f}->{row['ece_recal']:.3f} "
            f"auroc={row['auroc_recal']:.3f} [{row['verdict']}]"
        )
        del readout

    base_probs = probs_by_variant["base"]
    base_row = results["base"]
    print()
    for name in args.variants:
        if name == "base":
            results[name]["drift"] = "reference"
            continue
        kl = symmetric_kl(probs_by_variant[name], base_probs)
        agreement = float(
            (probs_by_variant[name].argmax(1) == base_probs.argmax(1)).mean()
        )
        mean_kl = float(kl.mean())
        results[name].update(
            {
                "top1_agreement_vs_base": agreement,
                "mean_sym_kl_vs_base": mean_kl,
                "p95_sym_kl_vs_base": float(np.percentile(kl, 95)),
                "auroc_drop_vs_base": base_row["auroc_recal"] - results[name]["auroc_recal"],
                "drift": classify(results[name], base_row, mean_kl),
            }
        )
        print(
            f"  {name} vs base: agreement={agreement:.3f} "
            f"symKL={mean_kl:.3f} -> {results[name]['drift']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "n_items": len(items),
                "manifest_fingerprint": manifest_fingerprint(items),
                "provenance": run_provenance(
                    {"variants": args.variants, "device": args.device, "dtype": args.dtype}
                ),
                "depth": args.depth,
                "variants": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
