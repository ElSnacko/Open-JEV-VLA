# Experiment brief: E0 and E1

What to run on the GPU box, in order, and what each outcome means. Both
experiments are cheap. E0 is CPU-feasible; a GPU just makes it minutes instead
of an hour.

Read `docs/VERIFICATION.md` first if you haven't. The short version: SmolVLA
keeps only the first 16 of SmolLM2-360M's 32 text layers, while freezing
`lm_head`, `text_model.norm` and the last two kept layers. So the decoder head
is intact and unused, sitting in every SmolVLA checkpoint, but it has never
been asked to read a half-depth residual stream. These two experiments ask it.

## Setup

```bash
git clone -b claude/dreamy-brahmagupta-ynbaoq https://github.com/ElSnacko/Open-JEV-VLA
cd Open-JEV-VLA
pip install -e ".[readout,dev]"
pytest -q                              # expect 25 passed, no weights needed
pip install -e ".[smolvla]"            # only needed for E1 and the manifest builder
```

## Step 1: build a manifest (~5 min)

E0 needs labelled multiple-choice items. `scripts/make_manifest.py` derives
them from a LeRobot dataset's own proprioception, so there is nothing to
annotate by hand:

```bash
python scripts/make_manifest.py --repo-id <a lerobot SO101 dataset> \
    --generator gripper --n 300 --out data/gripper.jsonl

python scripts/make_manifest.py --repo-id <same> \
    --generator phase --n 400 --out data/phase.jsonl
```

`gripper` (open vs closed, from the gripper joint) is the sanity question: a
readout that fails it fails everything. `phase` (early / midway / late, from
position in the episode) is the closest free proxy to the real target, task
progress monitoring.

**Eyeball the first ten frames before running anything.** Check that the label
matches what you see, and that the printed label balance isn't lopsided. Use
demonstration episodes, not failed rollouts: `phase` labels assume the episode
succeeded, otherwise you are training a stopwatch rather than a progress
monitor.

## Step 2: smoke test (~2 min)

Do not start with 400 items. The readout path has never executed, so the first
run is debugging:

```bash
head -20 data/gripper.jsonl > data/smoke.jsonl
python experiments/e0_depth_ablation.py --manifest data/smoke.jsonl \
    --depths 32 --device cuda --out results/smoke.json
```

What you want to see: a printed `[leading_space=..., mass=...]` line with mass
well above 0.05, and an accuracy clearly above 0.5 at full depth. Full depth
is the control. **If the full 32-layer model can't answer the question, the
question is broken, not the hypothesis.** Fix the prompt or the labels before
touching truncation.

Things likely to break on that first run, in rough order of probability:

1. `apply_chat_template` / image-token handling differs by transformers
   version. Symptom: a processor error, or image tokens not matching. Fix in
   `OptionReadout.build_prompt`.
2. Option markers not single tokens. The code raises deliberately rather than
   silently comparing incomparable probabilities. Try `leading_space=True` or
   different markers.
3. `_text_model` can't find the decoder stack. Symptom: `AttributeError` on
   truncation. Add the right attribute path to `jev/readout.py:_text_model`.
4. `mass` near zero at full depth. The model isn't answering in A/B/C format
   at all. Prompt problem. Rephrase before concluding anything.

## Step 3: E0, the depth ablation (~20 min on GPU)

```bash
python experiments/e0_depth_ablation.py --manifest data/gripper.jsonl \
    --depths 8 12 16 20 24 32 --device cuda --out results/e0_gripper.json
python experiments/e0_depth_ablation.py --manifest data/phase.jsonl \
    --depths 8 12 16 20 24 32 --device cuda --out results/e0_phase.json
```

Nothing here touches SmolVLA. It chops the *base* SmolVLM2 to each depth the
same way LeRobot does and measures what the frozen head can still decode. The
depths either side of 16 matter as much as 16 itself: a smooth decay tells a
different story from a cliff.

Decision rule, pre-registered in the script docstring, fixed before you look:

| verdict | condition at depth 16 | what it means |
|---|---|---|
| `FORMAT_FAILURE` | option mass < 0.05 | prompt bug, not a result. Fix and rerun. |
| `KILL` | accuracy ≤ chance + 2se, or recalibrated AUROC ≤ 0.55 | the readout cannot live on SmolVLA's backbone. Stop; write it up as a negative and pivot to a parallel full-depth VLM. |
| `SURVIVES` | clearly above chance and AUROC ≥ 0.65 | go to E1. |
| `INCONCLUSIVE` | between the two | more items, or a sharper question. |

The single most interesting outcome is accuracy holding while `ece_raw` blows
up and `ece_recal` comes back down. That would say truncation costs
*calibration*, not *content*, and that post-hoc scaling buys it back. That is a
clean result and nobody has it.

## Step 4: E1, the finetuning drift (~30 min on GPU)

```bash
python experiments/e1_finetune_drift.py --manifest data/phase.jsonl \
    --variants base smolvla libero --device cuda --out results/e1_phase.json
```

Same items, same depth, three weight sources: base SmolVLM2 truncated to 16,
`lerobot/smolvla_base`, and `lerobot/smolvla_libero` (more action training, so
the comparison has a gradient rather than two points).

| drift label | what it means |
|---|---|
| `INTACT` | readout can live on the policy. Best case. |
| `RECALIBRABLE` | accuracy holds, calibration degrades, post-hoc scaling restores it. Also good: calibrate per checkpoint. |
| `MOVED-BUT-OK` | distribution shifted a lot, ranking survived. Never transfer a threshold across checkpoints. |
| `DEGRADED` | readout cannot survive action finetuning. This is the architecture decision the whole line of work needs. Publish it. |

## What to record

Mostly automatic now. Every results file carries a `provenance` block (library
versions, GPU, repo commit, whether the working tree was dirty) and a
`manifest_fingerprint` hashing the questions, options, labels and frames.

**Two runs with different fingerprints are not comparable**, whatever the
filenames say. That is the failure mode to watch: tweak the prompt, rerun, then
compare calibration numbers as though nothing changed. Check the fingerprints
match before you put two runs in the same table.

`results/*.json` is tracked by git, so commit and push them. The frames under
`data/` are not; rebuild those from the manifest script.

Still worth noting by hand: the dataset repo id and revision you passed to
`make_manifest.py`, and anything you changed in the code mid-session that the
commit hash would not reflect.

## What E0 and E1 do not tell you

The probability the readout recovers is P(next word is "A"), not P(the action
is safe). The action expert and the language head share a trunk but were
trained for different things, and nothing forces them to agree. E0 shows the
mechanism functions; E1 shows action training did not break it; neither shows
the answers track action quality. That is E2. See `CONTEXT.md` section 2b
before writing up an encouraging E0 as more than a prerequisite.

## Why E2 is not in this brief

E2 (does the calibrated gate actually catch failures in closed loop) needs
rollouts, failure annotation and real GPU time. It is only worth paying for if
E0 and E1 survive. Two afternoons of E0 and E1 decide whether to spend that
month.
