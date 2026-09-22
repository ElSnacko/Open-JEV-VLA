# Context, caveats and the architectural argument

Everything a future reader (or a future session) needs before touching the
experiments. Read `VERIFICATION.md` for the prior-art check and
`EXPERIMENTS.md` for what to run.

---

## 1. How SmolVLA actually produces an action, and why there is no softmax

This is the load-bearing fact, and the natural assumption about it is wrong.

A reasonable intuition says: the model must pick an action, picking means
choosing from options, choosing means a softmax somewhere. **That intuition
holds for one family of VLA and not the other.**

**Discrete / autoregressive VLAs** (OpenVLA and relatives) quantise each action
dimension into bins, typically 256 of them, and emit the bin indices as
ordinary tokens through the language head. There is a real softmax over a real
vocabulary, so there are real per-action probabilities you can calibrate. That
is exactly what Zollo & Zemel (2507.17383) read.

**Continuous / flow-matching VLAs** (SmolVLA, π0, π0.5) never pick anything.
They regress the numbers directly. A joint angle is a real value, not an item
in a list, so nothing is being selected and nothing needs normalising.

SmolVLA's actual control path (`modeling_smolvla.py`):

```
images + language + state
  -> embed_prefix()                      builds the multimodal prefix
  -> vlm_with_expert.forward()           16 VLM layers, produces a KV cache
  -> [ denoise_step() x num_steps=10 ]   action expert cross-attends to that cache
       suffix_out = outputs_embeds[1]
       v_t = self.action_out_proj(suffix_out)      <- nn.Linear, no softmax
  -> euler_integrate(...)                integrates the velocity field
  -> action chunk, shape (batch, chunk_size=50, action_dim)
```

`action_out_proj` is `nn.Linear(expert_hidden_size, max_action_dim)`
(`modeling_smolvla.py:512`). It emits a velocity vector for a flow-matching
ODE. Start from Gaussian noise, take ten Euler steps along the predicted
velocity field, arrive at a chunk of 50 continuous actions. No categorical
distribution appears at any point.

So `lm_head` is not "unused because the roboticists were careless". It is
structurally unreachable: the action path leaves the language stack at the
hidden states and never returns to the vocabulary.

## 2. Why that makes SmolVLA the sharpest target, not the weakest

The usual framing of this project is "add one more uncertainty signal to the
pile". For SmolVLA that framing undersells it.

**SmolVLA has no probability anywhere in its control path.** Not a weak one, or
a poorly calibrated one. None. There is nothing to calibrate, because there is
nothing normalised to calibrate. Every published uncertainty method for this
class of policy therefore has to *manufacture* a signal:

- train a head on hidden states (SAFE 2506.09937, VLAConf 2605.29605),
- sample K times and measure disagreement (the test-time-scaling family that
  BOKBO 2605.30660 attacks),
- perturb activations and measure spread (2606.20754),
- conformalise the continuous action outputs directly (FabriVLA 2607.08575).

Each of those needs training data, extra forward passes, or both.

The frozen `lm_head` is the one place in the checkpoint where a genuine,
pretrained, normalised probability distribution already exists. It costs one
extra forward pass and no training whatsoever. **That is the whole idea.** The
only question is whether it still works after the stack it reads from has been
cut in half, which is E0.

## 3. What SmolVLA does to the VLM, precisely

Two separate operations, easy to conflate, and only one of them is a problem.

**Truncation (the problem).** `smolvlm_with_expert.py:102-105`:

```python
if num_vlm_layers > 0:
    self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
```

with `num_vlm_layers: int = 16` (`configuration_smolvla.py:96`). The backbone
is `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, whose text decoder is
SmolLM2-360M: **32 layers**. Half the stack is deleted. `lm_head` was trained
to decode layer 32 and would now be fed layer 16.

**Freezing (the gift).** `smolvlm_with_expert.py:163-174`:

```python
frozen_layers = ["lm_head", "text_model.norm.weight"]
for layer in last_layers:                      # the last one or two kept layers
    frozen_layers.append(f"text_model.layers.{layer}.")
```

The head, the final norm, and the layers immediately feeding them are the
original pretrained weights, untouched by action training. Everything below has
moved.

So the readout path is: frozen-ish top, moved bottom, half the depth. Nobody
has measured what comes out of it.

## 4. Caveats that apply to every number in this repository

**Nothing involving weights has ever run.** This repo was developed in a
container with no GPU, with `huggingface.co` and `arxiv.org` blocked at the
egress proxy. `jev/calibration.py` and `jev/evaluation.py` are tested (25
tests, including empirical verification of conformal coverage). `jev/readout.py`
and `scripts/make_manifest.py` are written against the real APIs, with the
LeRobot source read directly to get the truncation and dataset calls right, but
have never executed. **Treat the first run as debugging, not as a result.**
`EXPERIMENTS.md` lists the four most likely breakages.

**The prior-art check could not open a single PDF.** Every paper claim in
`VERIFICATION.md` rests on search-engine summaries. Titles, arXiv IDs and
author lists are reliable at that level. Specific numbers and method details
are not. Before investing past E0, spot check the SmolVLA paper (2506.01844)
and INSIGHT (2510.01389) appendices on a machine with arXiv access: a depth or
logit-lens ablation could plausibly be sitting in either.

**"Nobody has done this" is absence of evidence.** It comes from instrumented
search, not proof. The specific five-way combination (semantic options + one
pass + raw option-token logits + calibration + runtime gate, on the policy's
own backbone) appears unclaimed. The general area is crowded and moving fast.

**The novelty window is narrow and closing.** Roughly a dozen papers on VLA
uncertainty, abstention and failure gating in the fourteen months to September
2026. Two public GitHub repos (`jimfable/mech-int-vla`,
`yoonsp02-dotcom/failure-aware-smolvla`) are on the exact SmolVLA + LIBERO
failure-prediction stack, both with commits in the week of 2026-09-15.

**SmolVLA is recent, not old.** Released 2025-06-02. About fifteen months as of
this writing. "Surely someone tested this by now" carries less weight than it
sounds like it should.

**The most likely reason this is unpublished is that it was tried and failed.**
That is the base rate for cheap obvious experiments with no paper behind them.
The second most likely is that the interpretability community already assumes
the answer: logit lens from intermediate layers is known to give degraded,
badly calibrated output that improves with depth, so "reads worse at 16 of 32"
is not a finding to them.

**But the logit-lens prior does not settle this question.** It is about
*accuracy*. The question here is about *recalibrated ranking quality for
gating*. Those come apart: a readout that is 70% accurate and wildly
overconfident is a failed logit lens and a serviceable gate, because a gate
needs only that the confidence ranking survives and that a threshold can be
fitted. That gap is why E0 is still worth an afternoon.

**Recorded prediction, made before any run.** E0 comes back above chance at
depth 16, with poor raw ECE that recalibration substantially repairs: the
`RECALIBRABLE` middle outcome. Real but modest. A `KILL` or an `INTACT` would
both be more informative than that, in opposite directions. Written down so the
result can embarrass it.

## 5. Evidence that runs against the design

Four independent results say a single static readout at a chunk boundary is the
weakest version of this idea. None of them kills it, since the option-token
readout is a different signal from action-token entropy or K-sample
disagreement, but the burden of proof has shifted.

- **INSIGHT** (2510.01389): modelling the temporal evolution of token-level
  uncertainty beats static sequence-level scores by a wide margin.
- **2604.20472**: episode-level temporal-difference calibration beats
  per-action confidence for predicting rollout success.
- **BOKBO** (2605.30660): policy-internal nonconformity scores correlate 0.98
  with the action-noise hyperparameter σ. They track perturbation, not safety.
- **`jimfable/mech-int-vla`**, commit 2026-09-18: internals beat output-only
  features on a locked test, but lost to a simulator-privileged baseline.

If E0 and E1 survive, build the temporal version alongside the static one in
E2, or you are reproducing a known negative.

## 6. Why the readout must not become a probe

There is a constant temptation, once the raw readout looks weak, to train a
small head on the layer-16 hidden states instead. Resist it, or at least name
it when you do.

A trained probe is SAFE and VLAConf. It needs labelled failures, it is specific
to the checkpoint and task distribution it was trained on, and it is already
published. The raw readout needs no training at all, which is its entire
advantage and the only reason this is a distinct contribution.

If E0 kills the raw readout, the honest move is to report that and pivot to a
parallel full-depth VLM, not to quietly become the probe paper.
