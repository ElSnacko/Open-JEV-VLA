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

## Follow-up: does layer 16 actually carry the signal? (linear probe)

The KILL verdict above is compatible with two different explanations: either
layer 16 has no usable signal at all, or it has the signal and `lm_head`
specifically can't read it. Only the second is consistent with SmolVLA
working as a policy, but that argument was indirect -- inferred from the arm
functioning, not measured. `experiments/probe_layer16.py` measures it
directly: a regularized linear probe on the exact hidden vector `lm_head` is
fed (captured with a forward hook, not recomputed independently, so this is
provably the same vector the KILL verdict was measured on), scored against a
permutation-null baseline (same pipeline, shuffled labels) since 960 features
on ~150 training items is an easy regime to fool yourself in without one.

This is explicitly not a revival of the zero-training premise -- see the
script docstring and `docs/CONTEXT.md` section 6 for why a trained probe is a
different category of contribution (SAFE/VLAConf territory), named as such
rather than smuggled in as a JEV readout.

| task | depth | probe acc | chance | null acc (shuffled labels) | z vs null | auroc |
|---|---|---|---|---|---|---|
| gripper | 16 | 0.833 | 0.500 | 0.483 ± 0.044 | 8.0 | 0.904 |
| gripper | 32 | 0.880 | 0.500 | 0.488 ± 0.043 | 9.1 | 0.913 |
| phase | 16 | 0.860 | 0.333 | 0.418 ± 0.029 | 15.0 | 0.949 |
| phase | 32 | 0.878 | 0.333 | 0.410 ± 0.030 | 15.4 | 0.957 |

(Full objects in `results/probe_gripper.json`, `results/probe_phase.json`.
`LogisticRegressionCV` picked small `C` -- heavy L2 regularization -- on
every run, which argues against the high accuracy being memorization: a
model forced toward simple decision boundaries still separates the classes
cleanly.)

**This resolves the ambiguity: it's the first explanation's opposite.** Layer
16 carries the signal, decodably, at 8-15 standard deviations above what the
identical pipeline achieves on shuffled labels. And depth barely matters to
the probe -- 0.833 vs 0.880 on gripper, 0.860 vs 0.878 on phase -- a few
points, not the cliff `lm_head` shows between depth 24 and 32. Put the two
results side by side and the finding sharpens considerably: **truncation
costs almost nothing to the information in the residual stream, and almost
everything to the frozen decoder's ability to read it.** The KILL isn't
"SmolVLA's backbone stops encoding task state at depth 16." It's "the one
specific decoder this project tried to reuse for free was never fit to read
that depth, and nothing else about the representation is broken."

That reframes the project's two live next steps from `docs/CONTEXT.md`
section 6: option (a), a parallel full-depth VLM, is now a harder sell -- the
information is sitting in the policy's own truncated stack, not missing from
it, which is exactly the situation a parallel model would fail to exploit.
Option (b), a trained probe, has empirical footing behind it for the first
time in this repo: this experiment is close to a proof of concept for it,
not just a plausibility argument.

## Follow-up: why does `lm_head` fail specifically? (norm-mismatch diagnostic)

The probe leaves the *mechanism* of the format cliff unexplained. The
residual stream is additive across layers (`x_16 = x_0 + Δ_1 + ... + Δ_16`),
so layer 16 lives in roughly the same coordinate system as layer 32, just
with fewer refinements -- the standard reason logit-lens-style readouts
usually degrade gradually with depth rather than falling off a cliff. Two
candidate explanations for why this one didn't:

1. **Distribution mismatch.** The frozen final norm (`LlamaRMSNorm`, verified
   directly, `experiments/diagnose_norm_mismatch.py`) has a *fixed*, learned
   per-dimension weight vector, calibrated to layer 32's activation
   statistics. RMSNorm's division step corrects for gross magnitude
   differences between depths automatically, but the fixed weight can't
   correct for *directional* differences -- dimensions that only take on
   their layer-32 character in layers 17-32 get scaled wrong. This is the
   textbook motivation for "tuned lens" over naive "logit lens."
2. **Missing information**, in the specific subspace `lm_head`'s fixed
   weight matrix reads -- distinct from *any* discriminating subspace, which
   is all the probe needed.

Tested (1) directly rather than arguing for it: captured the exact hidden
vector fed to `lm_head` at both depth 16 and depth 32 (644 items pooled
across both manifests), computed each dimension's mean/std at both depths,
applied a simple moment-matching affine correction to the depth-16 vectors
(no training -- first and second moments only), and ran the *real* frozen
norm + `lm_head` on the corrected vectors.

| condition | gripper mass | gripper top token (n=300) | phase mass | phase top token (n=344) |
|---|---|---|---|---|
| raw depth-16 | 0.0000 | `'oriously'`, **every item** | 0.0000 | `'oriously'`, **every item** |
| moment-matched depth-16 | 0.873 | `' B'`, every item | 0.999 | `' B'`, every item |
| true depth-32 | 0.965 | `' A'` (298/300) | 0.998 | `' B'`, every item |

Full objects in `results/norm_mismatch_diagnostic.json`. Two things worth
separating:

**The format collapse at raw depth-16 is total, not partial** -- the exact
same single token, `'oriously'`, for all 644 items across both manifests.
Not noisy-but-varied garbage; a complete, item-independent collapse.

**A training-free statistics-only correction recovers format almost
completely** -- mass jumps from 0.0000 to 0.873-0.999, matching or nearly
matching the true depth-32 ceiling. This is about as direct a confirmation
of explanation (1) as this kind of test produces: nothing about the
*information* changed between the raw and corrected vectors, only its
scale and offset per dimension, and that alone was enough.

**But the corrected output is still constant per task** -- always `' B'`,
regardless of which of the 644 different images it was given. That is not a
new problem introduced by the correction: true depth-32 shows the identical
pattern (`' A'` for 298/300 gripper items, `' B'` for all 344 phase items),
which is the content-blindness already reported above. A first/second-moment
correction has no mechanism to fix that, because it can't reconstruct
whatever computation layers 17-32 would have contributed -- it only
rescales what's already in the depth-16 vector.

**So the KILL verdict has two independent causes, not one:**

- The format cliff (depths <= 24) is a fixed-norm/distribution-mismatch
  artifact, and a cheap, training-free correction resolves it almost
  completely. This is good news for a probe-style path forward: it says the
  representation itself doesn't need "fixing," just a different, trained
  read-out (which is exactly what a linear probe is).
- Content-blindness is separate, deeper, present even at genuine full depth,
  and this diagnostic does not touch it. It is not a truncation artifact at
  all -- whatever causes it, fixing truncation would not fix it.

This was a crude, moment-matching stand-in for a real tuned lens (fit by
gradient descent against next-token loss, not just two moments) -- a
negative result here would rule out only the simplest version of
explanation (1). The result wasn't negative, so that caveat matters less
than it would have, but a real tuned lens would still be the sharper version
of this check if the exact mechanism needs pinning down further.

## Follow-up: is there a rotational component beyond simple rescaling?

The diagonal correction above only rescales each of the 960 dimensions
independently -- it can't mix dimensions together. If depth-16 and depth-32
genuinely encoded information in a *different arrangement* of axes (not just
different per-axis scale), a diagonal-only fix should fail where a full
linear map succeeds, since reorganizing information requires mixing
dimensions. `experiments/tuned_lens_check.py` tests this directly: a
ridge-regularized full affine map (`y = Wx + b`, 960x960, CV-selected alpha,
fit and evaluated with a proper train/held-out-test split so the ~922k
parameters can't just memorize the ~300 training items) compared against the
diagonal correction, both scored on the same held-out items.

| task | diagonal mass | diagonal top token | ridge tuned lens mass | ridge top token | true depth-32 top token |
|---|---|---|---|---|---|
| gripper (151 test) | 0.874 | `' B'` (151/151) | **0.967** | **`' A'` (151/151)** | `' A'` (151/151) |
| phase (171 test) | 0.999 | `' B'` (171/171) | 0.998 | `' B'` (171/171) | `' B'` (171/171) |

Full object in `results/tuned_lens_check.json` (ridge alpha selected by CV:
10; train R^2 against depth-32 targets: 0.988).

**Task-dependent, and the evidence is asymmetric -- worth stating precisely
rather than as a clean two-way split.** On `gripper`, the full map does
something the diagonal correction is structurally incapable of: it doesn't
just recover more mass, it flips which constant answer comes out, from
`' B'` (matching nothing) to `' A'` (exactly matching true depth-32, on all
151 of 151 held-out items). A per-axis rescaling cannot flip which direction
wins; only mixing dimensions can. That's a complete, deterministic result on
the entire test set -- **a confirmed rotational component for this task,**
not a marginal trend.

On `phase`, the honest statement is narrower than "no rotational component."
Diagonal correction already reaches mass 0.999, against a true-depth-32
ceiling of 0.998 -- there is no headroom left for a rotational fix to
demonstrate a gain, whether or not one exists. **Absence of evidence, not
evidence of absence:** `phase` doesn't rule out a rotational component, it
saturates before one could be detected. Any task where the diagonal
correction already reaches the depth-32 ceiling will look this way regardless
of the true geometry underneath it.

**This changes the mechanistic picture, not the content-blindness finding.**
The rotation-corrected gripper output is exactly as constant across all 151
different images as before -- it now constant-matches the *correct* (and
equally content-blind) depth-32 answer instead of a different wrong one.
Whatever the geometric relationship between depth-16 and depth-32 turns out
to be, content-blindness lives downstream of it and is untouched by either
correction.

## What would change this conclusion

This used one 500M backbone (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
SmolVLA's actual base), one dataset (`lerobot/svla_so101_pickplace`), and two
generated-label tasks. Worth checking before treating any of this as final:

- Whether the `lm_head` mass cliff is specific to this base model's
  `lm_head`/norm pairing, or a general property of decoder-only VLM logit
  lenses at ~50% depth (the literature says degraded-but-present at
  intermediate depth, not a hard zero -- `docs/CONTEXT.md` section 4 flags
  this as worth spot checking against the SmolVLA and INSIGHT papers
  directly).
- Whether a larger SmolVLM2 checkpoint (2.2B) shows the same cliff shape at
  the equivalent relative depth, which would separate "half-depth kills
  logit-lens readouts on this model family" from "500M is just too small for
  this to work at all."
- Whether other SO101/SO100 datasets replicate the `lm_head` format cliff
  and the probe's success, or whether either is somehow specific to this
  dataset's visual distribution.
- Whether the probe result holds on the real `smolvla_base`/`smolvla_libero`
  checkpoints' own hidden states (this ran on the base-truncated VLM only,
  matching E0's "base" condition; E1 showed the real checkpoints track the
  base model closely on the `lm_head` side, but that hasn't been checked on
  the probe side).
- Whether the probe still separates classes on genuinely held-out episodes
  (train/test items here come from a 50/50 split of the same 344-frame pool,
  which can share an episode across the split; an episode-level split would
  rule out the probe keying on incidental per-episode cues like lighting or
  clutter rather than gripper/phase state itself).

None of these were run here; they are the next cheapest things if anyone
wants to push past this result rather than accept it.
