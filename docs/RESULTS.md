# E0 and E1 results

Run 2026-09-22 on an RTX 5090, the first time anything in this repo touched
real weights. `docs/CONTEXT.md` and `docs/EXPERIMENTS.md` predicted a range of
outcomes from `KILL` to `INTACT`; the actual result is more decisive than any
of them: **KILL, on two independent grounds, confirmed on the real SmolVLA
checkpoints.**

## Bug found before any result counted

`autodetect_leading_space` was called once per script, on the *last* iteration
of the depth/variant loop, but nothing propagated its finding to the readouts
already built for earlier iterations. Every depth in E0 except the deepest,
and every variant in E1, was silently scored with the default
`leading_space=False` tokenization instead of whatever the model actually
emits -- indistinguishable, in the output, from a genuine format failure at
that depth. Fixed by probing once at the reference depth/variant before the
loop and passing that setting into every subsequent `OptionReadout`
(`experiments/e0_depth_ablation.py`, `experiments/e1_finetune_drift.py`,
commit `1e7c8ca`). All numbers below are post-fix.

## E0: depth ablation, `lerobot/svla_so101_pickplace`

Two manifests built from that dataset's own proprioception (`gripper`, n=300,
label balance 149/151; `phase`, n=344, label balance 105/89/150). Both
eyeballed before running anything: the gripper labels match the jaw state in
the frame, and the phase labels match object position (frame 32 at
fraction 0.107: block on the table; frame 299 at fraction 1.0: block placed in
the tray).

| depth | gripper mass | gripper acc | gripper auroc | phase mass | phase acc | phase auroc |
|---|---|---|---|---|---|---|
| 8  | 0.0000 | 0.527 | 0.769 | 0.0000 | 0.320 | 0.491 |
| 12 | 0.0000 | 0.527 | 0.697 | 0.0000 | 0.320 | 0.493 |
| 16 | 0.0000 | 0.527 | 0.640 | 0.0000 | 0.320 | 0.496 |
| 20 | 0.0000 | 0.527 | 0.577 | 0.0000 | 0.320 | 0.542 |
| 24 | 0.0000 | 0.527 | 0.691 | 0.0000 | 0.320 | 0.472 |
| 32 | 0.9997 | 0.527 | 0.744 | 0.999 | 0.256 | 0.418 |

(chance = 0.500 for gripper, 0.333 for phase; full result objects, including
ECE and the conformal gate stats, in `results/e0_gripper.json` and
`results/e0_phase.json`.)

**This is a cliff, not a decay.** In-option mass is indistinguishable from
zero at every depth from 8 through 24, on both tasks, and jumps to ~1.0 only
at the untruncated 32. Manually inspecting the top-5 predicted tokens at
depths 8, 16 and 24 (both `leading_space` settings) confirms this is not a
borderline case: the model's highest-probability continuations are unrelated
vocabulary fragments (`' neighb'`, `'firehose'`, `'oriously'`, `'numerable'`,
`'Auschwitz'`) -- not near-miss letters, not a different single-token
preference, incoherent output. `docs/EXPERIMENTS.md` flagged "a smooth decay
tells a different story from a cliff" as the thing worth watching for. It's a
cliff.

At depth 32, format recovers (mass ~1.0) but accuracy does not: 0.527 on a
0.500-chance binary task, 0.256 on a 0.333-chance three-way task -- at or
below chance on both, and AUROC of 0.418 on `phase` is worse than a coin
flip. Manual probing (varying option order, varying prompt directiveness,
disabling SmolVLM2's default 13-tile image splitting) traced this to a
content-independent constant answer: the model picks the same option
regardless of which image it's shown, tracking the option's *wording*
(`open`, `midway through`) rather than the image, across every phrasing
tried. A free-generation sanity check on the same frames (no MCQA constraint,
no truncation) showed the base model *can* verbally identify gripper state
correctly, so the capability exists somewhere in the model -- it just doesn't
surface through this single-token readout, on this checkpoint, on this
manifest.

Identical accuracy across every depth on a given task (0.527 for all six
gripper depths, 0.320 for five of six phase depths) confirms the argmax is
depth-invariant: truncation changes how sharply the head can even reach the
option tokens, not which one it prefers once it gets there.

Verdict at depth 16 (the pre-registered decision rule, `jev/evaluation.py`):
**`FORMAT_FAILURE`** on both manifests. Depth 32, run through the same rule,
resolves to **`KILL`** (content-blind despite well-formatted output).

## E1: the real checkpoints, not just truncated base weights

E0 truncates the *base* SmolVLM2's own layers. SmolVLA's actual first 14 of
its 16 kept layers are action-finetuned, not merely deleted -- a genuinely
different condition worth checking empirically rather than assuming it fails
the same way. Depth 16, `phase` manifest, `lerobot/smolvla_base` and
`lerobot/smolvla_libero`:

| variant | mass | acc | auroc | agreement vs base | sym KL vs base |
|---|---|---|---|---|---|
| base (truncated SmolVLM2) | 0.000 | 0.320 | 0.537 | -- | -- |
| smolvla_base | (format failure) | 0.314 | 0.502 | 0.991 | 0.001 |
| smolvla_libero | (format failure) | 0.378 | 0.439 | 0.863 | 0.066 |

Full object in `results/e1_phase.json`. Both real checkpoints reproduce
`FORMAT_FAILURE`, and both track the truncated-base model's token-level
answers closely (99.1% and 86.3% top-1 agreement, symmetric KL near zero).
Action finetuning did not move the readout, in either direction, because
there was no working readout for it to move: the mechanism was already dead
before finetuning touched it. `smolvla_libero`'s lower agreement (86.3% vs
99.1%) is the only place LIBERO training shows up at all, and it's noise
around an already-broken baseline, not a signal worth reading into.

## Verdict

**KILL**, on two independent grounds, holding across two tasks, six depths,
and three checkpoints (base-truncated, `smolvla_base`, `smolvla_libero`):

1. The frozen `lm_head` + final norm cannot produce a coherent single-token
   answer from any residual stream shallower than the full 32 layers --
   including SmolVLA's actual, action-finetuned 16-layer stack, not just a
   synthetically truncated stand-in for it.
2. Where it *can* produce a coherent answer (depth 32 only, which SmolVLA
   never runs at), the answer does not condition on the image.

Per `docs/CONTEXT.md` section 6: this is the point to report the negative and
pivot, not to quietly retrain a probe on layer-16 hidden states and call it
the same contribution. E2 (closed-loop gating) is not worth running -- there
is no working readout for it to gate with, at the depth SmolVLA actually
uses. The two live options are (a) a parallel full-depth VLM run alongside
the policy, off its own backbone, at the cost this project was trying to
avoid, or (b) a trained probe on layer-16 hidden states, which is SAFE /
VLAConf territory and a different, already-published contribution -- name it
as that if pursued.

## What would change this conclusion

This used one 500M backbone (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
SmolVLA's actual base), one dataset (`lerobot/svla_so101_pickplace`), and two
generated-label tasks. Worth checking before treating this as final:

- Whether the mass cliff is specific to this base model's `lm_head`/norm
  pairing, or a general property of decoder-only VLM logit lenses at ~50%
  depth (the literature says degraded-but-present at intermediate depth, not
  a hard zero -- `docs/CONTEXT.md` section 4 flags this as worth spot
  checking against the SmolVLA and INSIGHT papers directly).
- Whether a larger SmolVLM2 checkpoint (2.2B) shows the same cliff shape at
  the equivalent relative depth, which would separate "half-depth kills
  logit-lens readouts on this model family" from "500M is just too small for
  this to work at all."
- Whether other SO101/SO100 datasets replicate both the format cliff and the
  content-blindness, or whether either is somehow specific to this dataset's
  visual distribution.

None of these were run here; they are the next cheapest things if anyone
wants to push past this result rather than accept it.
