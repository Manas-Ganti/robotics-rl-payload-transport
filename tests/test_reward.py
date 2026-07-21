"""Reward term edge cases. Runs WITHOUT Isaac Sim -- mock state only."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from env.reward import (
    RewardConfig,
    RewardState,
    TermSpec,
    compute_reward,
    term_action_rate,
    term_collision,
    term_energy_under_load,
    term_goal_reached,
    term_jerk,
    term_progress_shaping,
    term_step_cost,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_cfg(**overrides) -> RewardConfig:
    """A v1 reward config with the project's default weights."""
    defaults = dict(
        goal_reached=TermSpec(True, 200.0),
        collision=TermSpec(True, 100.0),
        step_cost=TermSpec(True, 0.05),
        progress_shaping=TermSpec(True, 10.0),
        progress_clip_per_step=1.0,
        payload_mass_norm_kg=5.0,
    )
    defaults.update(overrides)
    return RewardConfig(**defaults)


def make_state(**overrides) -> RewardState:
    defaults = dict(
        prev_distance=5.0,
        curr_distance=5.0,
        reached_goal=False,
        collided=False,
        action=np.array([0.0, 0.0]),
        prev_action=np.array([0.0, 0.0]),
        lin_vel=0.0,
        prev_lin_vel=0.0,
        payload_mass=3.0,
        dt=0.02,
    )
    defaults.update(overrides)
    return RewardState(**defaults)


# ---------------------------------------------------------------------------
# Individual terms
# ---------------------------------------------------------------------------
class TestGoalReached:
    def test_bonus_is_positive_on_success(self):
        assert term_goal_reached(make_state(reached_goal=True), TermSpec(True, 200.0)) == 200.0

    def test_zero_when_not_reached(self):
        assert term_goal_reached(make_state(reached_goal=False), TermSpec(True, 200.0)) == 0.0

    def test_zero_when_disabled(self):
        assert term_goal_reached(make_state(reached_goal=True), TermSpec(False, 200.0)) == 0.0


class TestCollision:
    def test_penalty_is_negative(self):
        """The sign is applied in code -- a positive config weight must still
        produce a negative contribution, or the policy learns to seek crashes."""
        assert term_collision(make_state(collided=True), TermSpec(True, 100.0)) == -100.0

    def test_zero_when_no_collision(self):
        assert term_collision(make_state(collided=False), TermSpec(True, 100.0)) == 0.0


class TestStepCost:
    def test_is_negative(self):
        assert term_step_cost(make_state(), TermSpec(True, 0.05)) == pytest.approx(-0.05)

    def test_broadcasts_to_batch_shape(self):
        """Must return one value per env, not a bare scalar -- otherwise it will
        not broadcast correctly against the other batched terms."""
        state = make_state(curr_distance=np.array([5.0, 3.0, 1.0]))
        result = term_step_cost(state, TermSpec(True, 0.5))
        assert np.asarray(result).shape == (3,)
        assert np.allclose(result, -0.5)


class TestProgressShaping:
    def test_positive_when_approaching_goal(self):
        state = make_state(prev_distance=5.0, curr_distance=4.5)
        assert term_progress_shaping(state, TermSpec(True, 10.0), 1.0) == pytest.approx(5.0)

    def test_negative_when_retreating(self):
        state = make_state(prev_distance=4.5, curr_distance=5.0)
        assert term_progress_shaping(state, TermSpec(True, 10.0), 1.0) == pytest.approx(-5.0)

    def test_zero_when_stationary(self):
        state = make_state(prev_distance=5.0, curr_distance=5.0)
        assert term_progress_shaping(state, TermSpec(True, 10.0), 1.0) == 0.0

    def test_clip_bounds_a_teleport_spike(self):
        """A reset or physics teleport can produce a huge distance delta. Without
        the clip that injects an enormous spurious reward, which is the classic
        cause of a great return curve alongside a terrible success rate."""
        state = make_state(prev_distance=100.0, curr_distance=1.0)
        assert term_progress_shaping(state, TermSpec(True, 10.0), 1.0) == pytest.approx(10.0)

    def test_clip_bounds_negative_spike_too(self):
        state = make_state(prev_distance=1.0, curr_distance=100.0)
        assert term_progress_shaping(state, TermSpec(True, 10.0), 1.0) == pytest.approx(-10.0)


class TestV2Terms:
    def test_action_rate_penalizes_churn(self):
        state = make_state(action=np.array([1.0, 1.0]), prev_action=np.array([0.0, 0.0]))
        assert term_action_rate(state, TermSpec(True, 0.05)) == pytest.approx(-0.1)

    def test_action_rate_zero_when_action_is_steady(self):
        state = make_state(action=np.array([0.5, 0.5]), prev_action=np.array([0.5, 0.5]))
        assert term_action_rate(state, TermSpec(True, 0.05)) == 0.0

    def test_action_rate_safe_when_inputs_missing(self):
        """v2 inputs are optional; a missing one must not crash a v1 run."""
        state = make_state(action=None, prev_action=None)
        assert term_action_rate(state, TermSpec(True, 0.05)) == 0.0

    def test_jerk_penalizes_acceleration(self):
        state = make_state(lin_vel=1.0, prev_lin_vel=0.0, dt=0.02)
        assert term_jerk(state, TermSpec(True, 0.02)) == pytest.approx(-0.02 * 50.0)

    def test_energy_scales_with_payload_mass(self):
        """A heavier payload must cost more energy at the same speed -- that
        coupling is the point of the transport-aware tier."""
        light = term_energy_under_load(
            make_state(lin_vel=1.0, payload_mass=1.0), TermSpec(True, 0.01), 5.0
        )
        heavy = term_energy_under_load(
            make_state(lin_vel=1.0, payload_mass=5.0), TermSpec(True, 0.01), 5.0
        )
        assert heavy < light < 0


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
class TestComputeReward:
    def test_components_always_present(self):
        """Disabled terms still appear (as 0.0) so W&B keys stay stable across
        ablations -- a vanishing chart series would look like a logging bug."""
        _, components = compute_reward(make_state(), make_cfg())
        for name in (
            "goal_reached", "collision", "step_cost", "progress_shaping",
            "action_rate", "jerk", "energy_under_load",
        ):
            assert name in components

    def test_total_equals_component_sum(self):
        state = make_state(prev_distance=5.0, curr_distance=4.5, reached_goal=True)
        total, components = compute_reward(state, make_cfg())
        assert total == pytest.approx(sum(np.asarray(v).sum() for v in components.values()))

    def test_success_outweighs_accumulated_step_cost(self):
        """Reaching the goal must beat the cost of the steps it took, or the
        optimal policy is to end the episode as fast as possible by any means."""
        state = make_state(reached_goal=True)
        total, _ = compute_reward(state, make_cfg())
        assert total > 0

    def test_collision_is_net_negative(self):
        state = make_state(collided=True)
        total, _ = compute_reward(state, make_cfg())
        assert total < 0

    def test_batched_state_produces_per_env_rewards(self):
        state = RewardState(
            prev_distance=np.array([5.0, 5.0, 5.0]),
            curr_distance=np.array([4.0, 5.0, 6.0]),
            reached_goal=np.array([False, False, False]),
            collided=np.array([False, False, False]),
        )
        total, _ = compute_reward(state, make_cfg())
        assert np.asarray(total).shape == (3,)
        # Approaching > stationary > retreating.
        assert total[0] > total[1] > total[2]

    def test_v2_terms_gated_off_by_phase_flag(self):
        """A v2 term enabled in YAML must stay inert while the phase flag is off,
        so the reward tier is controlled by exactly one switch."""
        cfg = RewardConfig.from_config(
            {
                "terms": {
                    "goal_reached": {"enabled": True, "weight": 200.0},
                    "collision": {"enabled": True, "weight": 100.0},
                    "step_cost": {"enabled": True, "weight": 0.5},
                    "progress_shaping": {"enabled": True, "weight": 10.0, "clip_per_step": 1.0},
                    "jerk": {"enabled": True, "weight": 0.02},
                },
                "version": "v1",
            },
            enable_reward_v2=False,
        )
        assert cfg.jerk.enabled is False
        assert cfg.jerk.active_weight == 0.0

    def test_v2_terms_activate_with_phase_flag(self):
        cfg = RewardConfig.from_config(
            {
                "terms": {
                    "goal_reached": {"enabled": True, "weight": 200.0},
                    "collision": {"enabled": True, "weight": 100.0},
                    "step_cost": {"enabled": True, "weight": 0.5},
                    "progress_shaping": {"enabled": True, "weight": 10.0, "clip_per_step": 1.0},
                    "jerk": {"enabled": True, "weight": 0.02},
                },
                "version": "v2",
            },
            enable_reward_v2=True,
        )
        assert cfg.jerk.enabled is True


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
class TestWeightGuardrails:
    def test_negative_weight_rejected(self):
        """Signs live in code, not config -- a negative weight would let a YAML
        typo flip a penalty into a bonus."""
        with pytest.raises(ValueError, match="non-negative"):
            TermSpec(True, -1.0)

    def test_warns_on_collision_as_escape(self):
        """Collision must cost more than a full episode of step costs, or
        crashing early becomes the cheapest way to end an episode."""
        cfg = make_cfg(collision=TermSpec(True, 1.0), step_cost=TermSpec(True, 0.5))
        with pytest.warns(UserWarning, match="COLLISION-AS-ESCAPE"):
            cfg.warn_on_degenerate_weights(episode_length_s=30.0, policy_dt=0.02)

    def test_warns_on_weak_terminal_signal(self):
        """If the journey pays more than the arrival, the policy learns to
        approach the goal without committing to entering it."""
        cfg = make_cfg(goal_reached=TermSpec(True, 5.0), progress_shaping=TermSpec(True, 10.0))
        with pytest.warns(UserWarning, match="WEAK TERMINAL SIGNAL"):
            cfg.warn_on_degenerate_weights(
                episode_length_s=30.0, policy_dt=0.02, max_goal_distance_m=14.14
            )

    def test_no_warning_on_shipped_defaults(self):
        """The weights shipped in configs/train.yaml must be free of every known
        pathology. This test is what keeps the defaults honest as they change."""
        cfg = make_cfg()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg.warn_on_degenerate_weights(
                episode_length_s=30.0, policy_dt=0.02, max_goal_distance_m=14.14
            )

    def test_rejects_zero_policy_dt(self):
        with pytest.raises(ValueError):
            make_cfg().warn_on_degenerate_weights(episode_length_s=30.0, policy_dt=0.0)
