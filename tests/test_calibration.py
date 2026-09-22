"""Tests for the calibration and gating math.

These run without torch, transformers, or model weights -- the point is that
the part of the pipeline that carries the statistical claims is validated
independently of whether a GPU is available.
"""

from __future__ import annotations

import numpy as np
import pytest

from jev.calibration import (
    apply_temperature,
    auroc,
    brier_score,
    calibrate_gate,
    conformal_threshold,
    expected_calibration_error,
    fit_temperature,
    fit_vector_scaling,
    lac_prediction_sets,
    nll,
    risk_coverage_curve,
    softmax,
)


def synthetic_logits(n, k, separation, temperature=1.0, seed=0, option_bias=None):
    """Draw logits that are calibrated by construction, then distort them.

    Latent logits mu are drawn with spread `separation`; the label is sampled
    from softmax(mu). That makes softmax(mu) the *exact* posterior, so the
    undistorted logits are perfectly calibrated and any calibration error we
    then measure comes from the distortion we applied, not from the generator.

    `temperature` over-sharpens (>1) the returned logits, standing in for an
    overconfident readout. `option_bias` adds a per-option prior, standing in
    for "the token 'A' is a likelier continuation than 'C'" -- a distortion a
    single temperature cannot remove.

    Larger `separation` means the options are more separable, i.e. a stronger
    readout; it is the knob the depth-sweep test uses to stand in for a
    shallower backbone.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, separation, size=(n, k))
    probs = softmax(mu)
    # vectorised categorical sampling from each row of `probs`
    labels = (probs.cumsum(axis=1) < rng.uniform(size=(n, 1))).sum(axis=1)
    labels = np.clip(labels, 0, k - 1)

    logits = mu * temperature
    if option_bias is not None:
        logits = logits + np.asarray(option_bias)
    return logits, labels


class TestMetrics:
    def test_softmax_normalises(self):
        p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
        assert np.allclose(p.sum(axis=1), 1.0)
        assert np.allclose(p[1], 1 / 3)

    def test_softmax_is_shift_invariant_and_stable(self):
        a = softmax(np.array([[1.0, 2.0, 3.0]]))
        b = softmax(np.array([[1001.0, 1002.0, 1003.0]]))
        assert np.allclose(a, b)
        assert np.isfinite(b).all()

    def test_ece_zero_for_perfectly_calibrated(self):
        # confidence p on the top class, correct exactly p of the time
        rng = np.random.default_rng(0)
        n = 40_000
        conf = rng.uniform(0.5, 1.0, size=n)
        probs = np.stack([conf, 1 - conf], axis=1)
        labels = (rng.uniform(size=n) > conf).astype(int)
        assert expected_calibration_error(probs, labels) < 0.02

    def test_ece_large_for_overconfident(self):
        n = 5_000
        probs = np.tile([0.99, 0.01], (n, 1))
        labels = np.zeros(n, dtype=int)
        labels[: n // 2] = 1  # only 50% correct while claiming 99%
        assert expected_calibration_error(probs, labels) > 0.4

    def test_auroc_perfect_and_chance(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert auroc(scores, np.array([False, False, True, True])) == 1.0
        assert auroc(scores, np.array([True, True, False, False])) == 0.0

    def test_auroc_all_ties_is_half(self):
        assert auroc(np.ones(10), np.array([True] * 5 + [False] * 5)) == 0.5

    def test_auroc_degenerate_mask_does_not_raise(self):
        assert auroc(np.arange(5.0), np.zeros(5, dtype=bool)) == 0.5

    def test_brier_and_nll_reward_the_truth(self):
        labels = np.array([0, 1])
        good = np.array([[0.9, 0.1], [0.1, 0.9]])
        bad = np.array([[0.1, 0.9], [0.9, 0.1]])
        assert brier_score(good, labels) < brier_score(bad, labels)
        assert nll(good, labels) < nll(bad, labels)

    def test_nll_does_not_overflow_on_zero_probability(self):
        assert np.isfinite(nll(np.array([[0.0, 1.0]]), np.array([0])))


class TestRecalibration:
    def test_temperature_recovers_known_distortion(self):
        # generate calibrated logits, then over-sharpen them by 3x
        base, labels = synthetic_logits(4_000, 3, separation=2.0, seed=1)
        sharpened = base * 3.0
        t = fit_temperature(sharpened, labels)
        # fitted temperature should undo roughly the 3x sharpening
        assert 2.0 < t < 4.5

    def test_temperature_improves_held_out_nll(self):
        logits, labels = synthetic_logits(3_000, 3, separation=1.5, temperature=4.0, seed=2)
        cal, test = slice(0, 1500), slice(1500, None)
        t = fit_temperature(logits[cal], labels[cal])
        before = nll(softmax(logits[test]), labels[test])
        after = nll(apply_temperature(logits[test], t), labels[test])
        assert after < before

    def test_temperature_near_one_when_already_calibrated(self):
        logits, labels = synthetic_logits(6_000, 3, separation=2.0, seed=3)
        assert 0.7 < fit_temperature(logits, labels) < 1.4

    def test_vector_scaling_removes_per_option_bias(self):
        # a strong prior favouring option A, uncorrelated with the label
        bias = [2.5, 0.0, 0.0]
        logits, labels = synthetic_logits(4_000, 3, separation=1.5, seed=4, option_bias=bias)
        cal, test = slice(0, 2000), slice(2000, None)
        w, b = fit_vector_scaling(logits[cal], labels[cal])
        # the fitted bias should push back against option A
        assert b[0] < b[1] and b[0] < b[2]
        before = nll(softmax(logits[test]), labels[test])
        after = nll(softmax(logits[test] * w + b), labels[test])
        assert after < before

    def test_single_temperature_cannot_fix_per_option_bias(self):
        """Why vector scaling is in the pipeline and not just temperature."""
        bias = [3.0, 0.0, 0.0]
        logits, labels = synthetic_logits(4_000, 3, separation=1.5, seed=5, option_bias=bias)
        cal, test = slice(0, 2000), slice(2000, None)
        t = fit_temperature(logits[cal], labels[cal])
        temp_only = nll(apply_temperature(logits[test], t), labels[test])
        w, b = fit_vector_scaling(logits[cal] / t, labels[cal])
        both = nll(softmax(logits[test] / t * w + b), labels[test])
        assert both < temp_only


class TestConformal:
    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
    def test_split_conformal_achieves_marginal_coverage(self, alpha):
        """The load-bearing guarantee: sets cover the truth >= 1-alpha."""
        rng = np.random.default_rng(7)
        coverages = []
        for trial in range(40):
            logits, labels = synthetic_logits(1_200, 3, separation=1.2, seed=100 + trial)
            idx = rng.permutation(len(labels))
            cal, test = idx[:600], idx[600:]
            probs = softmax(logits)
            qhat = conformal_threshold(1.0 - probs[cal, labels[cal]], alpha)
            sets = lac_prediction_sets(probs[test], qhat)
            coverages.append(sets[np.arange(len(test)), labels[test]].mean())
        # marginal coverage, averaged over calibration draws
        assert np.mean(coverages) >= 1.0 - alpha - 0.02

    def test_threshold_is_infinite_when_calibration_set_too_small(self):
        # n=5 cannot certify alpha=0.05: ceil(6*0.95)/5 = 1.2 > 1
        assert conformal_threshold(np.linspace(0, 1, 5), alpha=0.05) == float("inf")

    def test_smaller_alpha_gives_larger_sets(self):
        logits, labels = synthetic_logits(2_000, 3, separation=1.0, seed=8)
        probs = softmax(logits)
        scores = 1.0 - probs[np.arange(len(labels)), labels]
        q_tight = conformal_threshold(scores, 0.3)
        q_loose = conformal_threshold(scores, 0.05)
        assert q_loose > q_tight
        assert lac_prediction_sets(probs, q_loose).sum() > lac_prediction_sets(probs, q_tight).sum()

    def test_empty_calibration_set_raises(self):
        with pytest.raises(ValueError):
            conformal_threshold(np.array([]), 0.1)


class TestGate:
    def test_gate_acts_only_on_confident_act_option(self):
        logits, labels = synthetic_logits(2_000, 3, separation=3.0, seed=9)
        gate = calibrate_gate(logits, labels, alpha=0.1, act_option=0)
        act, set_size = gate.decide(logits)
        # every acted-on item must be a singleton pointing at option 0
        probs = gate.probabilities(logits)
        assert set(np.unique(set_size[act])) <= {1}
        assert (probs[act].argmax(1) == 0).all()

    def test_gate_abstains_more_as_signal_degrades(self):
        """Simulates the E0 depth sweep without a model in the loop.

        Weaker separation stands in for a shallower backbone. The gate should
        respond by acting less often, not by acting wrongly.
        """
        act_rates = []
        for sep in (4.0, 2.0, 1.0, 0.25):
            logits, labels = synthetic_logits(8_000, 3, separation=sep, seed=11)
            cal, test = slice(0, 4000), slice(4000, None)
            gate = calibrate_gate(logits[cal], labels[cal], alpha=0.1, act_option=0)
            act, _ = gate.decide(logits[test])
            act_rates.append(float(act.mean()))
        assert act_rates == sorted(act_rates, reverse=True), act_rates
        # a near-signal-free readout should almost never fire the gate
        assert act_rates[-1] < 0.05, act_rates

    def test_gate_round_trips_through_its_dataclass_fields(self):
        logits, labels = synthetic_logits(1_000, 3, separation=2.0, seed=12)
        gate = calibrate_gate(logits, labels, alpha=0.1)
        assert gate.n_calibration == 1_000
        assert gate.scale.shape == (3,) and gate.bias.shape == (3,)
        assert gate.probabilities(logits[0]).shape == (1, 3)


class TestRiskCoverage:
    def test_selective_prediction_lowers_risk_at_low_coverage(self):
        logits, labels = synthetic_logits(4_000, 3, separation=1.5, seed=13)
        probs = softmax(logits)
        correct = probs.argmax(1) == labels
        _, coverage, risk = risk_coverage_curve(probs.max(1), correct)
        full = risk[coverage.argmax()]
        selective = risk[coverage < 0.3].mean()
        assert selective < full

    def test_risk_coverage_is_flat_for_uninformative_scores(self):
        rng = np.random.default_rng(14)
        correct = rng.uniform(size=4_000) < 0.6
        scores = rng.uniform(size=4_000)  # independent of correctness
        _, coverage, risk = risk_coverage_curve(scores, correct)
        mid = (coverage > 0.2) & (coverage < 0.9)
        assert risk[mid].std() < 0.05
