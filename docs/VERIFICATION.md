# Verification of `jev-readout-vla-robotics-brief-2026-09-22.md`

Checked 2026-09-22. **Read the caveat before the findings.**

## Caveat on method

`arxiv.org`, `huggingface.co` and `alphaxiv.org` are all blocked by this
container's egress proxy. I could not open a single PDF. Everything below
marked *(search)* comes from search-engine summaries of those pages, not from
reading them. Titles, IDs and author lists are reliable at that level; specific
numbers and method details are not, and anything load-bearing should be
re-checked against the actual paper on a machine with network access.

What I *could* verify directly: the LeRobot source (cloned at `b64fe1e`,
2026-09-22) and two third-party GitHub repositories. Those findings are marked
*(source)* and are solid.

---

## 1. The citations hold up. The negative claim does not.

Every arXiv ID the brief flagged as unverified resolves to a real paper with
roughly the claimed content *(search)*:

| ID | Resolves to | Brief's characterisation |
|---|---|---|
| 2507.17383 | Confidence Calibration in Vision-Language-Action Models (Zollo & Zemel) | accurate, but it covers OpenVLA, **MolmoAct, UniVLA, NORA** — not OpenVLA-OFT |
| 2605.29605 | VLAConf: Calibrated Task-Success Confidence for VLAs (Huang et al.) | accurate |
| 2604.16677 | ReconVLA: Uncertainty-Guided and Failure-Aware VLA (UT Arlington) | accurate |
| 2607.08575 | FabriVLA: Lightweight VLA with Conformal Action Chunk Uncertainty | accurate |
| 2506.09937 | SAFE: Multitask Failure Detection for VLAs | accurate |

So the brief is not built on hallucinated references. That was the first thing
worth ruling out and it is ruled out.

The problem is the other direction: **the brief's 20+ search queries missed a
lot.** Papers on the same target that it never mentions *(search)*:

- **INSIGHT** (2510.01389, Yale, Oct 2025) — per-token entropy, log-prob and
  Dirichlet aleatoric/epistemic estimates off π0-FAST, fed to transformer
  classifiers that trigger asking for help. This is the closest published
  thing to "port KnowNo's help-trigger idea onto a VLA", and it explicitly
  argues KnowNo does not transfer directly. The brief's central framing is
  partly somebody else's paper from eleven months ago.
- **BOKBO** (2605.30660, May 2026) — "the first conformal abstention layer for
  K-sample VLA inference", finite-sample guarantees on executed-violation rate,
  global and Mondrian per-task variants.
- **VLA-FAIL** (2606.21386) — last-layer Mahalanobis + action-chunk consistency,
  both with calibrated thresholds.
- **ActProbe** (2606.08508) — action-space probe for early failure detection.
- **2606.20754** — perturbation-based epistemic uncertainty for VLA failure
  detection on LIBERO / LIBERO-PRO.
- **CheckVLA** (2607.26789) — execution-time verification with an
  action-conditioned world model.
- **TOKENPROB** — aggregated action-token probability as a cheap reference signal.
- **2604.11662** — "Hidden Failures in Robustness: Why Supervised Uncertainty
  Quantification Needs Better Evaluation".

That is roughly a dozen papers on VLA uncertainty, abstention and failure
gating in about fourteen months. The field is not empty and it is not slow.

## 2. Two people are already working on the exact MVP stack

Both public on GitHub, both committed within the last five days *(source,
cloned and read)*:

- **`jimfable/mech-int-vla`** — "Pre-registered mechanistic interpretability
  study of failure prediction in SmolVLA on LIBERO". Full pre-registration
  (`PREREG.md`), locked test set, content-addressed artifacts, 473-test suite.
  Its latest commit (2026-09-18) reports: *internals beat output-only features
  on the locked test, but fell short of M1* — the simulator-privileged
  baseline. That is a partial negative result for internals-based failure
  prediction on this exact stack, from someone running it properly.
- **`yoonsp02-dotcom/failure-aware-smolvla`** — SmolVLA + LIBERO-Spatial
  evaluation harness, failure-targeted data selection, first smoke test
  2026-09-17.

Neither is doing the semantic option-token readout specifically. Both occupy
the ground around it.

## 3. A technical blocker the brief missed entirely

**SmolVLA keeps only the first 16 of its VLM's 32 text layers.** *(source)*

`lerobot/policies/smolvla/smolvlm_with_expert.py:102-105`:

```python
if num_vlm_layers > 0:
    print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
    self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
```

with `num_vlm_layers: int = 16` in `configuration_smolvla.py:96`, and the
backbone being `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (line 84) whose
text decoder is SmolLM2-360M — **32 layers** *(search)*.

This breaks MVP option (b) as written. "A JEV-style second cheap forward of the
same SmolVLM2 backbone ... read A/B/C token distribution via `output_scores`"
assumes the policy's backbone is a language model that can decode. It is a
half-depth stump. The `lm_head` and `text_model.norm` weights *do* survive —
SmolVLA explicitly freezes them, along with the last two kept layers
(`smolvlm_with_expert.py:163-174`) — but they were trained to read a 32-layer
residual stream and would be fed a 16-layer one.

So there is no "just run the backbone as a VLM" path. There are three paths,
and they are not equivalent:

1. Truncated stack → frozen norm → frozen `lm_head`. Cheap, genuinely reads the
   policy's own state. **Completely unmeasured.**
2. A parallel full-depth SmolVLM2 as the readout model. Works for sure, but it
   is a second model judging the scene — which is the AHA / I-FailSense
   free-text mainstream the brief dismisses, just with option tokens instead of
   prose. Much weaker novelty claim.
3. A trained probe on layer-16 hidden states. Also works, but that is SAFE and
   VLAConf, not a JEV readout.

Path 1 is the only one where the brief's framing survives, and nobody has
tested whether it functions at all. **That is the experiment.** See
`experiments/e0_depth_ablation.py`.

Minor related correction: the brief says "SmolVLM2-400M". It is the 500M
Video-Instruct checkpoint; ~450M is the assembled SmolVLA policy.

## 4. Expansion opportunity #2 is mostly already answered

The brief calls calibration-transfer — "does action-token finetuning warp the
backbone's semantic option-token distribution?" — something "nobody has
measured". The phenomenon is well documented *(search)*:

- **Knowledge Insulation** (2505.23705) — don't propagate the continuous action
  expert's gradient into the VLM, precisely because it damages the backbone.
- **AEGIS** (2604.16067) — "action fine-tuning injects a qualitatively alien
  training signal that overwrites VLM representations destructively"; naive
  finetuning "causes significant, steady degradation in VQA holdout loss within
  1,500 steps". Fix: frozen reference copy + distillation loss.
- **Actions as Language** (2509.22195) — actions as natural language + LoRA to
  avoid catastrophic forgetting.
- **2511.06619** — "How Do VLAs Effectively Inherit from VLMs?"

The *specific* measurement — option-token calibration, not VQA loss — is still
open. But the direction is known, and the architectural conclusion the brief
wanted this experiment to deliver is already implied: don't put a semantic
readout downstream of unconstrained MSE action gradients. Keep the experiment
if you want the number; don't sell it as an open question.

## 5. The evidence trend runs against the MVP's design

The MVP gates on a thresholded max-prob at chunk boundaries: one static score,
one decision point. Four independent results point the other way:

- **INSIGHT** *(search)*: modelling the temporal evolution of token-level
  uncertainty with transformers gives "far greater predictive power than static
  sequence-level scores".
- **2604.20472**, cited in the brief itself: episode-level temporal-difference
  calibration beats per-action confidence for predicting rollout success.
- **BOKBO** *(search)*: policy-internal nonconformity scores and K-sample
  disagreement correlate **0.98 with the action-noise hyperparameter σ** — they
  track perturbation, not safety. That is a direct structural attack on naive
  internal-confidence gating.
- **`mech-int-vla`** *(source)*: internals beat outputs but lost to a
  privileged non-internal baseline on a locked test.

None of these kills the option-token readout — it is a different signal from
action-token entropy or K-sample disagreement, and BOKBO's σ-correlation
finding is specific to perturbation-based sampling. But the burden of proof
has shifted. A single static readout is the design the literature keeps finding
weakest, so if you build it, build the temporal version alongside it or you are
reproducing a known negative.

---

## What survives

The five-way combination — semantic decision-shaped options + one pass + raw
option-token logits + calibration + runtime gate, **on the policy's own
backbone** — still appears unclaimed. The brief is right about that.

But the claim is much narrower than "port KnowNo to VLAs", INSIGHT already took
the obvious framing, two public repos are circling the same stack, and the
specific instantiation on SmolVLA is blocked by a depth-truncation problem the
brief did not know about.

"First to market" on the general idea is gone. What is available is being first
with a *measured answer* to one sharp question, and that answer is interesting
whichever way it lands.

## Recommended sequence

1. **E0 — depth ablation** (hours, one 500M model, CPU-feasible). Can a
   16-layer-truncated SmolVLM2 + original head answer A/B/C above chance, and
   does recalibration make it usable? Pre-registered kill criteria are in the
   script. No robot, no rollouts, no GPU rental.
2. **E1 — finetuning drift** (needs the SmolVLA checkpoint, still no robot).
   Same items, readout through SmolVLA's actual finetuned weights vs the base
   model's first 16 layers. Isolates action-finetuning damage from truncation
   damage.
3. **E2 — gating** (needs rollouts and a GPU). Only worth paying for if E0 and
   E1 survive.

E0 and E1 cost an afternoon between them and either kill the premise or hand
you the one result nobody has. That is the fail-fast version.

## Sources

Papers *(all via search summaries; none opened directly)*:
[2507.17383](https://arxiv.org/abs/2507.17383),
[2605.29605](https://arxiv.org/abs/2605.29605),
[2604.16677](https://arxiv.org/abs/2604.16677),
[2607.08575](https://arxiv.org/abs/2607.08575),
[2506.09937](https://arxiv.org/abs/2506.09937),
[2510.01389 INSIGHT](https://arxiv.org/abs/2510.01389),
[2605.30660 BOKBO](https://arxiv.org/abs/2605.30660),
[2606.21386 VLA-FAIL](https://arxiv.org/html/2606.21386),
[2606.08508 ActProbe](https://arxiv.org/pdf/2606.08508),
[2606.20754](https://arxiv.org/pdf/2606.20754),
[2607.26789 CheckVLA](https://arxiv.org/abs/2607.26789),
[2604.11662](https://arxiv.org/abs/2604.11662),
[2505.23705 Knowledge Insulation](https://arxiv.org/pdf/2505.23705),
[2604.16067 AEGIS](https://arxiv.org/pdf/2604.16067),
[2509.22195 Actions as Language](https://arxiv.org/abs/2509.22195),
[2511.06619](https://arxiv.org/pdf/2511.06619).

Code *(read directly)*:
[huggingface/lerobot](https://github.com/huggingface/lerobot) @ `b64fe1e`,
[jimfable/mech-int-vla](https://github.com/jimfable/mech-int-vla),
[yoonsp02-dotcom/failure-aware-smolvla](https://github.com/yoonsp02-dotcom/failure-aware-smolvla),
[HuggingFaceTB/SmolLM2-360M config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json).
