"""Tests for run provenance and manifest fingerprinting.

Both exist so that two results files can be compared honestly months later.
They are only useful if they never crash a run and never silently call two
different setups identical.
"""

from __future__ import annotations

from jev.evaluation import manifest_fingerprint, run_provenance


def item(question="is it on track?", options=None, label=0, image="a.png"):
    return {
        "question": question,
        "options": options or ["yes", "no"],
        "label": label,
        "image": image,
    }


class TestProvenance:
    def test_survives_missing_optional_dependencies(self):
        """torch and lerobot are absent here; provenance must still return."""
        prov = run_provenance()
        assert prov["python"]
        assert prov["timestamp_utc"].endswith("Z")
        assert "torch_version" in prov  # present as a key even when None

    def test_extra_fields_are_merged(self):
        prov = run_provenance({"model_id": "x", "device": "cuda"})
        assert prov["model_id"] == "x" and prov["device"] == "cuda"

    def test_extra_can_override_a_captured_field(self):
        assert run_provenance({"python": "pinned"})["python"] == "pinned"


class TestFingerprint:
    def test_identical_manifests_match(self):
        a = [item(), item(label=1, image="b.png")]
        b = [item(), item(label=1, image="b.png")]
        assert manifest_fingerprint(a) == manifest_fingerprint(b)

    def test_changed_prompt_changes_the_fingerprint(self):
        """The failure mode this exists to catch."""
        base = [item()]
        tweaked = [item(question="is it on track? answer with one letter.")]
        assert manifest_fingerprint(base) != manifest_fingerprint(tweaked)

    def test_changed_options_or_labels_change_it(self):
        base = [item()]
        assert manifest_fingerprint([item(options=["on track", "blocked"])]) != manifest_fingerprint(base)
        assert manifest_fingerprint([item(label=1)]) != manifest_fingerprint(base)

    def test_ignores_fields_that_do_not_affect_comparability(self):
        a = [item()]
        b = [dict(item(), image_path="/absolute/elsewhere/a.png")]
        assert manifest_fingerprint(a) == manifest_fingerprint(b)

    def test_item_order_matters(self):
        """Order changes which half is the calibration split, so it counts."""
        a = [item(image="a.png"), item(image="b.png")]
        b = [item(image="b.png"), item(image="a.png")]
        assert manifest_fingerprint(a) != manifest_fingerprint(b)
