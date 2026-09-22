# Open-JEV-VLA

Calibrated option-token readout and gating for vision-language-action policies.

**Status: pre-experiment.** The calibration and gating math is implemented and
tested. The readout is implemented against the real LeRobot/transformers APIs
but has not been run against weights — see *Why nothing has run yet* below.

## The question

KnowNo (Ren et al., CoRL 2023) asks a text LLM planner a multiple-choice
question, reads the raw likelihoods of the option tokens in a single decode
pass, conformal-calibrates them, and gates ask-for-help against autonomous
execution. Every ingredient of that recipe exists separately on VLA models. The
assembled thing does not appear to.

The obvious port is: ask the VLA's own backbone "is this plan on track / blocked
/ unsafe", read the A/B/C token logits at the decision position, calibrate,
gate at chunk boundaries.

**That port is blocked on SmolVLA and nobody has said so.** LeRobot builds
SmolVLA by keeping only the first 16 of its VLM's 32 text layers. The `lm_head`
and final norm survive (SmolVLA freezes both) but were trained to decode a
full-depth residual stream. Whether they still decode anything useful from a
half-depth one is unmeasured, and it decides whether the readout can live on
the policy at all.

See [`docs/VERIFICATION.md`](docs/VERIFICATION.md) for the full prior-art check,
including a dozen papers and two active GitHub repos on adjacent ground.

## The experiments

| | what it settles | cost |
|---|---|---|
| **E0** depth ablation | can a 16-layer-truncated SmolVLM2 + original head answer A/B/C above chance, and does recalibration make it gate-worthy? | one 500M model, CPU-feasible, hours |
| **E1** finetuning drift | does action finetuning move the readout, beyond what truncation already costs? | + the SmolVLA checkpoint, still no robot |
| **E2** gating | does the calibrated gate actually catch failures in closed loop? | rollouts, GPU |

E0 is the kill test and it is deliberately the cheapest thing in the
repository. Its decision rule is pre-registered in the script docstring: fix
the thresholds before you look at the numbers.

```bash
python experiments/e0_depth_ablation.py \
    --manifest data/mcqa.jsonl \
    --depths 8 12 16 20 24 32 \
    --out results/e0.json
```

E0 does not need robot frames to be informative. If the truncated head cannot
answer A/B/C on *any* labelled image-MCQA set, it will not answer them on
gripper frames either, and that is a one-afternoon answer. Robot frames sharpen
the result; they are not needed to kill it.

## Layout

```
jev/calibration.py   temperature + vector scaling, split conformal, ECE/Brier/
                     NLL/AUROC, risk-coverage, the gate.  NumPy only.
jev/readout.py       single-pass option-token readout; layer truncation that
                     mirrors LeRobot's exactly; SmolVLA checkpoint loader.
experiments/         E0 and the ones that follow it.
tests/               25 tests over the calibration math, incl. empirical
                     verification of the conformal coverage guarantee.
```

`jev/calibration.py` is NumPy-only on purpose: the part of the pipeline that
carries the statistical claims runs and is tested without torch, weights, or a
GPU.

```bash
pip install -e ".[dev]" && pytest -q      # 25 passed
pip install -e ".[readout]"               # + torch, transformers, pillow
```

## Manifest format

One JSON object per line, image paths relative to the manifest:

```json
{"image": "frames/0001.png",
 "question": "Is the gripper on track to grasp the red block?",
 "options": ["on track", "blocked", "unsafe"],
 "label": 0}
```

## Why nothing has run yet

This was developed in a container with no GPU and with `huggingface.co` and
`arxiv.org` blocked at the egress proxy. No model weights could be downloaded,
so the readout path is written against the APIs but unexecuted; treat its first
run as debugging, not as a result. The calibration layer has no such excuse and
is tested.

## Prior art you should read before adding to this

Not a complete list — the full one with links is in `docs/VERIFICATION.md`.

- **INSIGHT** (2510.01389) — token-level uncertainty → help triggers on
  π0-FAST. Closest published framing to this one, and it finds that *temporal*
  modelling beats static per-decision scores.
- **BOKBO** (2605.30660) — conformal abstention for VLA policies. Finds
  policy-internal confidence proxies correlate 0.98 with the action-noise
  hyperparameter, i.e. they track perturbation rather than safety.
- **SAFE** (2506.09937) — learned head on VLA hidden states + conformal
  threshold. The probe-shaped alternative to this readout.
- **Zollo & Zemel** (2507.17383) — action-token calibration, and the source of
  the recalibration recipe used here.
