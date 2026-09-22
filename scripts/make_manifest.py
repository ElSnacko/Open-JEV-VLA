#!/usr/bin/env python3
"""Build an MCQA manifest for E0/E1 from a LeRobot dataset.

The friction in E0 is not compute, it is getting labelled multiple-choice
items. Hand-labelling failure frames needs rollouts and annotation. This
sidesteps that: a LeRobot dataset already carries proprioception alongside
every frame, so some questions about the image have ground truth for free.

Two generators, both label-free to run:

  gripper   "Is the gripper open or closed?"  Label from the gripper joint in
            observation.state. Binary, unambiguous, and plainly visible in the
            image. This is the sanity question: a readout that cannot answer
            it is not going to answer anything harder.

  phase     "Is this early / midway / late in the task?"  Label from the
            frame's position within its episode. Three options, and it is the
            closest free proxy to the actual target (task-progress monitoring)
            because the visual evidence is real: object displaced, gripper
            near or far, scene rearranged.

Caveats, both load-bearing:

  * `phase` labels are only meaningful on *successful demonstration* episodes.
    In a demo, "late in the episode" really does mean "late in the task". On a
    failed rollout it does not, and using this generator there would train you
    to predict elapsed time rather than progress.
  * `gripper` assumes the last dimension of observation.state is the gripper
    and that larger means more open. True for SO100/SO101. Check the printed
    histogram before trusting a run on any other embodiment.

Unvalidated: written against the LeRobot dataset API but never executed, as
this was developed without dataset access. Eyeball the first ten items.

Usage:
    python scripts/make_manifest.py --repo-id lerobot/svla_so101_pickplace \
        --generator phase --n 400 --out data/phase.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def to_pil(frame_value):
    """LeRobot image features come back as CHW float tensors in [0, 1]."""
    from PIL import Image

    arr = np.asarray(frame_value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr)


def pick_image_key(dataset, explicit: str | None) -> str:
    if explicit:
        return explicit
    keys = [
        k
        for k, spec in dataset.features.items()
        if spec.get("dtype") in {"image", "video"} or k.startswith("observation.image")
    ]
    if not keys:
        raise SystemExit("no image feature found; pass --image-key")
    # prefer a wrist-free overhead/front view: it shows the whole scene
    for preferred in ("top", "front", "overhead", "cam_high", "image"):
        for k in keys:
            if preferred in k:
                return k
    return keys[0]


def gripper_items(dataset, indices, gripper_dim: int, open_threshold: float | None):
    """Binary open/closed from the gripper joint."""
    states = np.array(
        [np.asarray(dataset[int(i)]["observation.state"]).reshape(-1)[gripper_dim] for i in indices]
    )
    # split at the median unless told otherwise: gripper positions are
    # typically bimodal, and the median lands between the two modes
    thr = float(np.median(states)) if open_threshold is None else open_threshold
    print(f"  gripper dim {gripper_dim}: min={states.min():.3f} "
          f"median={thr:.3f} max={states.max():.3f}")
    print(f"  histogram: {np.histogram(states, bins=10)[0].tolist()}")

    out = []
    for i, s in zip(indices, states, strict=True):
        out.append(
            (
                int(i),
                {
                    "question": "Look at the robot gripper. Is it open or closed?",
                    "options": ["open", "closed"],
                    "label": 0 if s > thr else 1,
                    "state_value": float(s),
                },
            )
        )
    return out


def phase_items(dataset, indices):
    """Early / midway / late, from position within the episode."""
    lengths: dict[int, int] = {}
    rows = []
    for i in indices:
        frame = dataset[int(i)]
        ep = int(np.asarray(frame["episode_index"]).item())
        fr = int(np.asarray(frame["frame_index"]).item())
        lengths[ep] = max(lengths.get(ep, 0), fr + 1)
        rows.append((int(i), ep, fr))

    out = []
    for i, ep, fr in rows:
        frac = fr / max(lengths[ep] - 1, 1)
        # drop the boundary bands: a frame at 0.34 is not cleanly "early",
        # and forcing a label there just adds noise the readout cannot fix
        if 0.28 < frac < 0.38 or 0.62 < frac < 0.72:
            continue
        label = 0 if frac <= 0.33 else (1 if frac <= 0.67 else 2)
        out.append(
            (
                i,
                {
                    "question": (
                        "This robot is partway through a manipulation task. "
                        "How far along is it?"
                    ),
                    "options": ["just started", "midway through", "nearly finished"],
                    "label": label,
                    "episode_fraction": round(float(frac), 4),
                },
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo-id", required=True, help="LeRobot dataset repo id")
    ap.add_argument("--generator", choices=["gripper", "phase"], default="phase")
    ap.add_argument("--n", type=int, default=400, help="frames to sample")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--image-key", default=None)
    ap.add_argument("--gripper-dim", type=int, default=-1)
    ap.add_argument("--open-threshold", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, episodes=args.episodes)
    image_key = pick_image_key(dataset, args.image_key)
    print(f"{args.repo_id}: {len(dataset)} frames, image key {image_key!r}")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.n, len(dataset)), replace=False)
    indices = np.sort(indices)

    if args.generator == "gripper":
        items = gripper_items(dataset, indices, args.gripper_dim, args.open_threshold)
    else:
        items = phase_items(dataset, indices)

    frames_dir = args.out.parent / (args.out.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.out.open("w") as fh:
        for idx, rec in items:
            path = frames_dir / f"{idx:07d}.png"
            to_pil(dataset[idx][image_key]).save(path)
            rec["image"] = str(path.relative_to(args.out.parent))
            fh.write(json.dumps(rec) + "\n")
            written += 1

    counts = np.bincount([r["label"] for _, r in items])
    print(f"wrote {written} items to {args.out}")
    print(f"label balance: {counts.tolist()} (chance = {1 / len(counts):.3f})")
    if counts.min() < 0.15 * counts.sum():
        print("WARNING: labels are badly imbalanced; accuracy will be misleading, "
              "read AUROC instead")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
