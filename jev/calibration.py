"""Calibration and gating math for option-token readouts.

NumPy only, on purpose: every function here runs and is tested without
torch, transformers, a GPU, or model weights. The readout side (jev.readout)
is the part that needs a model; this part can be validated anywhere.

Convention used throughout:
    logits  (n, k)  raw option-token logits, k options
    labels  (n,)    integer index of the correct option
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "softmax",
    "expected_calibration_error",
    "brier_score",
    "nll",
    "auroc",
    "fit_temperature",
    "apply_temperature",
    "fit_vector_scaling",
    "conformal_threshold",
    "lac_prediction_sets",
    "risk_coverage_curve",
    "GateCalibration",
    "calibrate_gate",
]


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# --------------------------------------------------------------------------
# calibration metrics
# --------------------------------------------------------------------------


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Top-label ECE with equal-width bins.

    This is the standard "confidence vs accuracy" ECE. It is a summary, not a
    guarantee: a model can have low ECE and still be useless for gating if its
    confidence does not separate the classes at all. Always read it next to
    AUROC.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins so that conf == 1.0 lands in the last bin
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    n = len(conf)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score (sum of squared error over the one-hot target)."""
    probs = np.asarray(probs, dtype=np.float64)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    p = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(p, eps, 1.0))))


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUROC with tie correction. `positive` is a boolean mask.

    Returns 0.5 when either class is empty, which keeps degenerate calibration
    splits from raising in the middle of a sweep.
    """
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------
# post-hoc recalibration
# --------------------------------------------------------------------------


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=np.float64) / float(temperature))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    lo: float = 0.02,
    hi: float = 50.0,
    tol: float = 1e-4,
) -> float:
    """Single-parameter temperature scaling by ternary search on held-out NLL.

    NLL as a function of log-temperature is unimodal for the standard softmax,
    so ternary search is enough and avoids a torch dependency for one scalar.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)

    def objective(t: float) -> float:
        return nll(apply_temperature(logits, t), labels)

    a, b = np.log(lo), np.log(hi)
    while b - a > tol:
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if objective(np.exp(m1)) < objective(np.exp(m2)):
            b = m2
        else:
            a = m1
    return float(np.exp((a + b) / 2.0))


def fit_vector_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    n_steps: int = 500,
    lr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-option scale and bias (w, b) fitted by full-batch gradient descent.

    Vector scaling is the multiclass analogue of Platt scaling. It matters here
    because option-token logits carry a per-letter prior ("A" is a more likely
    continuation than "C" for reasons that have nothing to do with the scene),
    and a single temperature cannot remove a per-letter bias.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    n, k = logits.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), labels] = 1.0

    w = np.ones(k)
    b = np.zeros(k)
    for _ in range(n_steps):
        p = softmax(logits * w + b)
        g = (p - onehot) / n
        w -= lr * np.sum(g * logits, axis=0)
        b -= lr * np.sum(g, axis=0)
    return w, b


# --------------------------------------------------------------------------
# conformal prediction
# --------------------------------------------------------------------------


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile with the finite-sample correction.

    Returns qhat such that, for an exchangeable test point, the prediction set
    {y : score(y) <= qhat} covers the truth with probability >= 1 - alpha.
    """
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    n = len(cal_scores)
    if n == 0:
        raise ValueError("empty calibration set")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        # too few calibration points to certify this alpha
        return float("inf")
    return float(np.quantile(cal_scores, level, method="higher"))


def lac_prediction_sets(
    probs: np.ndarray, qhat: float
) -> np.ndarray:
    """Least-Ambiguous-Set classifier: include option y iff 1 - p(y) <= qhat.

    This is the score KnowNo uses. Set size is the ambiguity signal: size 1
    means act, size > 1 means the options are not separated, size 0 cannot
    happen for qhat >= 1 - max(p).
    """
    probs = np.asarray(probs, dtype=np.float64)
    return (1.0 - probs) <= qhat


def risk_coverage_curve(
    scores: np.ndarray, correct: np.ndarray, n_points: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Selective-prediction curve: sweep the abstention threshold.

    Returns (thresholds, coverage, risk) where coverage is the fraction of
    items acted on and risk is the error rate among those. This is the
    "threshold knob" property: it is the thing that has to hold for a gate to
    be useful, and it is independent of whether ECE looks good.
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    lo, hi = float(scores.min()), float(scores.max())
    thresholds = np.linspace(lo, hi, n_points)

    coverage = np.empty(n_points)
    risk = np.empty(n_points)
    for i, t in enumerate(thresholds):
        acted = scores >= t
        coverage[i] = acted.mean()
        risk[i] = float(1.0 - correct[acted].mean()) if acted.any() else 0.0
    return thresholds, coverage, risk


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


@dataclass
class GateCalibration:
    """Everything needed to turn raw option logits into an act/stop decision."""

    temperature: float
    scale: np.ndarray
    bias: np.ndarray
    qhat: float
    alpha: float
    act_option: int
    n_calibration: int

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        logits = np.atleast_2d(np.asarray(logits, dtype=np.float64))
        return softmax((logits / self.temperature) * self.scale + self.bias)

    def decide(self, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (act, set_size).

        act is True only when the conformal prediction set is the singleton
        {act_option}: the readout is both confident and pointing at "proceed".
        An ambiguous set (size > 1) is a stop, which is the conservative
        reading and the one that carries the coverage guarantee.
        """
        probs = self.probabilities(logits)
        sets = lac_prediction_sets(probs, self.qhat)
        set_size = sets.sum(axis=1)
        act = (set_size == 1) & sets[:, self.act_option]
        return act, set_size


def calibrate_gate(
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float = 0.1,
    act_option: int = 0,
    vector_scaling: bool = True,
) -> GateCalibration:
    """Fit recalibration and the conformal threshold on one calibration split.

    Both stages are fitted on the same split, which is standard practice but
    does mean the coverage guarantee is approximate rather than exact. Use
    `split_calibration_set` upstream if you need the clean two-split version.
    """
    cal_logits = np.asarray(cal_logits, dtype=np.float64)
    cal_labels = np.asarray(cal_labels)

    temperature = fit_temperature(cal_logits, cal_labels)
    scaled = cal_logits / temperature
    if vector_scaling:
        w, b = fit_vector_scaling(scaled, cal_labels)
    else:
        w, b = np.ones(cal_logits.shape[1]), np.zeros(cal_logits.shape[1])

    probs = softmax(scaled * w + b)
    true_probs = probs[np.arange(len(cal_labels)), cal_labels]
    qhat = conformal_threshold(1.0 - true_probs, alpha)

    return GateCalibration(
        temperature=temperature,
        scale=w,
        bias=b,
        qhat=qhat,
        alpha=alpha,
        act_option=act_option,
        n_calibration=len(cal_labels),
    )
