"""Shared evaluation for the readout experiments.

Split out of E0 so E1 scores its weight variants with exactly the same code
path. If the two experiments measured things slightly differently, the
comparison between them would be worthless.
"""

from __future__ import annotations

import math

import numpy as np

from jev.calibration import (
    auroc,
    brier_score,
    calibrate_gate,
    expected_calibration_error,
    nll,
    softmax,
)

# pre-registered thresholds, shared by E0 and E1
CHANCE_MARGIN_SE = 2.0
AUROC_KILL = 0.55
AUROC_PASS = 0.65
FORMAT_FLOOR = 0.05


def collect_logits(readout, items) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the readout over every manifest item. Returns (logits, labels, mass)."""
    from PIL import Image

    rows, masses = [], []
    for rec in items:
        with Image.open(rec["image_path"]) as im:
            res = readout.score(im.convert("RGB"), rec["question"], rec["options"])
        rows.append(res.logits)
        masses.append(res.meta["in_option_mass"])
    labels = np.array([r["label"] for r in items])
    return np.stack(rows), labels, float(np.mean(masses))


def evaluate(logits: np.ndarray, labels: np.ndarray, seed: int = 0, alpha: float = 0.1) -> dict:
    """Split-half: fit recalibration on one half, report on the other.

    Fitting and reporting on the same items would make every recalibrated
    number optimistic, and recalibration is exactly what is under test here.
    """
    n, k = logits.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = n // 2
    cal, test = perm[:cut], perm[cut:]

    raw = softmax(logits[test])
    correct_raw = raw.argmax(1) == labels[test]

    gate = calibrate_gate(logits[cal], labels[cal], alpha=alpha)
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
        # does confidence rank correctness? the gating-relevant number
        "auroc_raw": auroc(raw.max(1), correct_raw),
        "auroc_recal": auroc(recal.max(1), correct_recal),
        "temperature": gate.temperature,
        "conformal_qhat": gate.qhat,
        "mean_set_size": float(set_size.mean()),
        "act_rate": float(act.mean()),
    }


def verdict(row: dict, in_option_mass: float) -> str:
    """Apply the pre-registered decision rule. See E0's docstring."""
    if in_option_mass < FORMAT_FLOOR:
        return "FORMAT_FAILURE"
    above_chance = row["accuracy"] > row["chance"] + CHANCE_MARGIN_SE * row["accuracy_se"]
    if not above_chance or row["auroc_recal"] <= AUROC_KILL:
        return "KILL"
    if row["auroc_recal"] >= AUROC_PASS:
        return "SURVIVES"
    return "INCONCLUSIVE"


def symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-item symmetric KL between two option distributions.

    Used in E1 to measure how far action finetuning moved the readout,
    independently of whether either version is any good.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return (p * np.log(p / q)).sum(axis=1) + (q * np.log(q / p)).sum(axis=1)


def load_manifest(path) -> list[dict]:
    """Read a JSONL manifest, resolving image paths relative to it."""
    import json
    import pathlib

    path = pathlib.Path(path)
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
