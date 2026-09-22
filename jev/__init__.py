"""Open-JEV-VLA: calibrated option-token readout and gating for VLA policies."""

from jev.calibration import (
    GateCalibration,
    auroc,
    brier_score,
    calibrate_gate,
    conformal_threshold,
    expected_calibration_error,
    fit_temperature,
    lac_prediction_sets,
    nll,
    risk_coverage_curve,
    softmax,
)

__version__ = "0.1.0"

__all__ = [
    "GateCalibration",
    "auroc",
    "brier_score",
    "calibrate_gate",
    "conformal_threshold",
    "expected_calibration_error",
    "fit_temperature",
    "lac_prediction_sets",
    "nll",
    "risk_coverage_curve",
    "softmax",
]
