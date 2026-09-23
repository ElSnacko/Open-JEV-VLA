# Open-JEV-VLA

Calibrated option-token readout and gating for vision-language-action policies.

**Status: E0 and E1 ran, both KILL — but a follow-up probe shows the readout
died for a specific, narrow reason, not because the information is gone.**
The frozen `lm_head` cannot decode a truncated residual stream (KILL, on two
tasks, six depths, and three checkpoints), but a trained linear probe on the
same layer-16 hidden states gets 83-88% accuracy, 8-15 standard deviations
above a permutation-null baseline. Truncation costs the frozen decoder almost
everything and the information itself almost nothing. See
[`docs/RESULTS.md`](docs/RESULTS.md) for the numbers and
[`docs/CONTEXT.md`](docs/CONTEXT.md) section 6 for what that means next. The
*Why nothing had run yet* section below is now history, kept for how the
predictions read before any weights were touched.

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

SmolVLA is also the sharpest place to ask it. It is a flow-matching policy:
its action path runs `action_out_proj`, a plain `nn.Linear` emitting a velocity
field, and integrates it. There is **no softmax anywhere in its control path**,
so there is no probability to calibrate. Every published uncertainty method for
this class of policy has to manufacture one, by training a head, sampling K
times, or perturbing activations. The frozen `lm_head` is the only pretrained
normalised distribution already sitting in the checkpoint, and it costs one
forward pass and no training.

Read in order:
[`docs/RESULTS.md`](docs/RESULTS.md) (what actually happened when this ran),
[`docs/CONTEXT.md`](docs/CONTEXT.md) (the architecture argument and every
caveat), [`docs/VERIFICATION.md`](docs/VERIFICATION.md) (prior-art check),
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (what was run, written before any
of it had been -- read it for the pre-registered decision rules, not as a
prediction of the outcome).

## The experiments

| | what it settles | result |
|---|---|---|
| **E0** depth ablation | can a 16-layer-truncated SmolVLM2 + original head answer A/B/C above chance, and does recalibration make it gate-worthy? | **KILL** — mass ≈0 at every depth ≤24, on both tasks |
| **E1** finetuning drift | does action finetuning move the readout, beyond what truncation already costs? | **KILL confirmed on the real checkpoints** — `smolvla_base`/`smolvla_libero` reproduce E0's format failure, 99%/86% token-agreement with the truncated base |
| **E2** gating | does the calibrated gate actually catch failures in closed loop? | not run — no working readout at SmolVLA's depth to gate with |

E0 was the kill test and it killed. Its decision rule was pre-registered in
the script docstring, fixed before any numbers existed:

```bash
python experiments/e0_depth_ablation.py \
    --manifest data/gripper.jsonl \
    --depths 8 12 16 20 24 32 \
    --device cuda --out results/e0_gripper.json
```

Full numbers, the leading-space bug found and fixed along the way, and what
would need to change to revisit this: [`docs/RESULTS.md`](docs/RESULTS.md).

## Layout

```
jev/calibration.py   temperature + vector scaling, split conformal, ECE/Brier/
                     NLL/AUROC, risk-coverage, the gate.  NumPy only.
jev/readout.py       single-pass option-token readout; layer truncation that
                     mirrors LeRobot's exactly; SmolVLA checkpoint loader.
experiments/         E0 and the ones that follow it.
tests/               33 tests over the calibration math, incl. empirical
                     verification of the conformal coverage guarantee.
```

`jev/calibration.py` is NumPy-only on purpose: the part of the pipeline that
carries the statistical claims runs and is tested without torch, weights, or a
GPU.

```bash
pip install -e ".[dev]" && pytest -q      # 33 passed
pip install -e ".[readout]"               # + torch, transformers, pillow
pip install -e ".[smolvla]"               # + lerobot, for E1 and the manifest builder
```

## Manifest format

One JSON object per line, image paths relative to the manifest:

```json
{"image": "frames/0001.png",
 "question": "Is the gripper on track to grasp the red block?",
 "options": ["on track", "blocked", "unsafe"],
 "label": 0}
```

## Why nothing had run, for a while

This was developed in a container with no GPU and with `huggingface.co` and
`arxiv.org` blocked at the egress proxy, so the readout path was written
against the real APIs but unexecuted, and every number in `docs/CONTEXT.md`
and `docs/EXPERIMENTS.md` was a prediction rather than a result. That changed
2026-09-22 on a machine with both a GPU and open egress: see
[`docs/RESULTS.md`](docs/RESULTS.md) for what actually happened. The
calibration layer never had this excuse -- it was NumPy-only and tested from
the start.

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
