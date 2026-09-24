"""Payload spec + inertia math. Runs WITHOUT Isaac Sim.

The payload is the study's dynamics axis, so the pure pieces that feed the
simulator -- which attach mode is configured, and the box inertia written
alongside every per-env mass -- are pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from env.config import load_train_config
from env.payload import ATTACH_MODES, PayloadSpec, box_inertia_diagonal

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"


def test_real_config_builds_a_deck_link_spec():
    spec = PayloadSpec.from_config(load_train_config(TRAIN_CONFIG), mass_kg=25.0)
    assert spec.attach_mode == "deck_link"
    assert spec.mass_kg == 25.0
    assert len(spec.size_m) == 3 and min(spec.size_m) > 0


def test_unknown_attach_mode_is_rejected():
    with pytest.raises(ValueError, match="attach_mode"):
        PayloadSpec(mass_kg=1.0, size_m=(0.4, 0.3, 0.2), offset_m=(0, 0, 0), attach_mode="glue")
    assert "deck_link" in ATTACH_MODES


def test_box_inertia_matches_closed_form():
    # 12 kg, 1 x 2 x 3 m: I = m/12 * (sum of squares of the other two sides)
    ixx, iyy, izz = box_inertia_diagonal(12.0, (1.0, 2.0, 3.0))
    assert (ixx, iyy, izz) == pytest.approx((13.0, 10.0, 5.0))


def test_box_inertia_scales_linearly_with_mass():
    """set_link_payload_mass relies on this: inertia is rewritten per mass."""
    light = box_inertia_diagonal(10.0, (0.4, 0.3, 0.2))
    heavy = box_inertia_diagonal(90.0, (0.4, 0.3, 0.2))
    assert [h / l for h, l in zip(heavy, light)] == pytest.approx([9.0, 9.0, 9.0])
