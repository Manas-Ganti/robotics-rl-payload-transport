"""Config schema + the sacred train/OOD split. Runs WITHOUT Isaac Sim.

The most important tests in the repository. If the train/OOD split is violated
and nothing catches it, the study measures nothing -- and the results will look
completely normal while doing so.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from env.config import (
    STUDY_AXES,
    Config,
    ConfigError,
    Range,
    assert_no_train_ood_overlap,
    assert_nominals_in_train_range,
    classify_point,
    extract_ranges,
    load_eval_config,
    load_train_config,
    load_yaml,
    validate_eval_config,
    validate_reward_config,
    validate_train_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"
EVAL_CONFIG = REPO_ROOT / "configs" / "eval_ood.yaml"


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------
class TestRange:
    def test_rejects_inverted_bounds(self):
        with pytest.raises(ConfigError, match="exceeds"):
            Range(10.0, 5.0)

    def test_contains_is_inclusive(self):
        r = Range(1.0, 5.0)
        assert r.contains(1.0) and r.contains(5.0) and r.contains(3.0)
        assert not r.contains(0.99) and not r.contains(5.01)

    def test_detects_overlap(self):
        assert Range(0.0, 5.0).overlaps(Range(4.0, 8.0))
        assert Range(4.0, 8.0).overlaps(Range(0.0, 5.0))

    def test_disjoint_ranges_do_not_overlap(self):
        assert not Range(0.0, 5.0).overlaps(Range(6.0, 8.0))

    def test_touching_endpoints_count_as_overlap(self):
        """[0,10] and [10,20] share the point 10.0, which would be both
        in-distribution and held-out. That ambiguity is exactly what the split
        exists to prevent, so touching must fail."""
        assert Range(0.0, 10.0).overlaps(Range(10.0, 20.0))

    def test_nested_range_overlaps(self):
        assert Range(0.0, 10.0).overlaps(Range(3.0, 4.0))

    def test_from_yaml_rejects_malformed(self):
        with pytest.raises(ConfigError, match="low, high"):
            Range.from_yaml([1.0, 2.0, 3.0], key="test")
        with pytest.raises(ConfigError):
            Range.from_yaml("not-a-range", key="test")


# ---------------------------------------------------------------------------
# THE SACRED SPLIT
# ---------------------------------------------------------------------------
class TestTrainOODSplit:
    def test_shipped_configs_have_no_overlap(self):
        """The configs in this repo must satisfy the split. This is THE test."""
        train_ranges = extract_ranges(
            load_yaml(TRAIN_CONFIG)["domain"]["train"], block_name="domain.train"
        )
        ood_ranges = extract_ranges(
            load_yaml(EVAL_CONFIG)["domain"]["ood"], block_name="domain.ood"
        )
        assert_no_train_ood_overlap(train_ranges, ood_ranges)

    def test_every_study_axis_is_split(self):
        """No axis may quietly go unsplit -- an unsplit axis is not being studied."""
        train_ranges = extract_ranges(
            load_yaml(TRAIN_CONFIG)["domain"]["train"], block_name="domain.train"
        )
        ood_ranges = extract_ranges(
            load_yaml(EVAL_CONFIG)["domain"]["ood"], block_name="domain.ood"
        )
        assert set(train_ranges) == set(STUDY_AXES)
        assert set(ood_ranges) == set(STUDY_AXES)

    def test_overlap_raises(self):
        with pytest.raises(ConfigError, match="TRAIN/OOD SPLIT VIOLATED"):
            assert_no_train_ood_overlap(
                {"payload_mass_kg": Range(1.0, 5.0)},
                {"payload_mass_kg": Range(4.0, 9.0)},
            )

    def test_error_names_the_offending_axis(self):
        with pytest.raises(ConfigError, match="payload_mass_kg"):
            assert_no_train_ood_overlap(
                {"payload_mass_kg": Range(1.0, 5.0)},
                {"payload_mass_kg": Range(4.0, 9.0)},
            )

    def test_missing_ood_axis_warns_but_passes(self):
        with pytest.warns(UserWarning, match="absent from OOD"):
            assert_no_train_ood_overlap(
                {"friction_coeff": Range(0.4, 0.9), "slope_angle_deg": Range(0.0, 10.0)},
                {"friction_coeff": Range(0.1, 0.3)},
            )

    def test_loading_a_contaminated_eval_config_fails(self, tmp_path):
        """End-to-end: a bad split must blow up at LOAD time, before any GPU
        time is spent producing meaningless numbers."""
        train_data = load_yaml(TRAIN_CONFIG)
        eval_data = load_yaml(EVAL_CONFIG)

        # Contaminate: make the OOD payload range overlap the training range.
        eval_data["domain"]["ood"]["payload_mass_kg"] = [4.0, 9.0]

        train_path = tmp_path / "train.yaml"
        eval_path = tmp_path / "eval.yaml"
        train_data["robot_config"] = str(REPO_ROOT / "configs" / "robot.yaml")
        eval_data["train_config"] = str(train_path)
        train_path.write_text(yaml.safe_dump(train_data), encoding="utf-8")
        eval_path.write_text(yaml.safe_dump(eval_data), encoding="utf-8")

        with pytest.raises(ConfigError, match="TRAIN/OOD SPLIT VIOLATED"):
            load_eval_config(eval_path)


class TestNominals:
    def test_shipped_nominals_are_in_distribution(self):
        """A nominal value outside the training range would confound every
        single-axis sweep -- degradation could no longer be attributed to the
        axis under test."""
        train_ranges = extract_ranges(
            load_yaml(TRAIN_CONFIG)["domain"]["train"], block_name="domain.train"
        )
        assert_nominals_in_train_range(load_yaml(EVAL_CONFIG)["nominal"], train_ranges)

    def test_out_of_range_nominal_raises(self):
        with pytest.raises(ConfigError, match="IN-DISTRIBUTION"):
            assert_nominals_in_train_range(
                {"payload_mass_kg": 8.0}, {"payload_mass_kg": Range(1.0, 5.0)}
            )

    def test_unknown_axis_raises(self):
        with pytest.raises(ConfigError, match="unknown axis"):
            assert_nominals_in_train_range(
                {"not_an_axis": 1.0}, {"payload_mass_kg": Range(1.0, 5.0)}
            )


class TestClassifyPoint:
    def test_labels_train_ood_and_gap(self):
        train_r, ood_r = Range(0.0, 10.0), Range(12.0, 20.0)
        assert classify_point(5.0, train_r, ood_r) == "train"
        assert classify_point(15.0, train_r, ood_r) == "ood"
        # Between the ranges: probes the boundary, but is neither trained-on nor
        # declared held-out. Must not be mislabelled as either.
        assert classify_point(11.0, train_r, ood_r) == "gap"

    def test_gap_when_no_ood_range(self):
        assert classify_point(15.0, Range(0.0, 10.0), None) == "gap"

    def test_every_shipped_grid_point_is_classifiable(self):
        """Each grid point must land in a labelled regime, and each enabled axis
        must include BOTH in-distribution and held-out points -- otherwise its
        retention metric has no baseline to divide by."""
        train_ranges = extract_ranges(
            load_yaml(TRAIN_CONFIG)["domain"]["train"], block_name="domain.train"
        )
        eval_data = load_yaml(EVAL_CONFIG)
        ood_ranges = extract_ranges(eval_data["domain"]["ood"], block_name="domain.ood")

        for axis, spec in eval_data["grid"].items():
            if not spec.get("enabled", False):
                continue
            regimes = {
                classify_point(float(v), train_ranges[axis], ood_ranges.get(axis))
                for v in spec["points"]
            }
            assert "train" in regimes, f"grid.{axis} has no in-distribution anchor points"
            assert "ood" in regimes, f"grid.{axis} has no held-out points"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
class TestSchemaValidation:
    def test_shipped_train_config_is_valid(self):
        validate_train_config(load_yaml(TRAIN_CONFIG))

    def test_shipped_eval_config_is_valid(self):
        validate_eval_config(load_yaml(EVAL_CONFIG))

    def test_shipped_train_config_loads(self):
        cfg = load_train_config(TRAIN_CONFIG)
        assert cfg.env.num_envs > 0
        assert cfg.robot.name  # robot.yaml was merged in

    def test_shipped_eval_config_loads(self):
        cfg = load_eval_config(EVAL_CONFIG)
        assert "train" in cfg.to_dict()

    def test_missing_required_key_raises(self):
        data = load_yaml(TRAIN_CONFIG)
        del data["reward"]
        with pytest.raises(ConfigError, match="missing required keys"):
            validate_train_config(data)

    def test_wrong_schema_version_raises(self):
        data = load_yaml(TRAIN_CONFIG)
        data["schema_version"] = 999
        with pytest.raises(ConfigError, match="schema_version"):
            validate_train_config(data)

    def test_unknown_study_axis_rejected(self):
        """A typo'd axis name must fail loudly rather than silently dropping a
        randomization axis from the entire study."""
        with pytest.raises(ConfigError, match="Unknown study axes"):
            extract_ranges({"payload_mass_kgg": [1.0, 5.0]}, block_name="domain.train")

    def test_missing_axis_rejected(self):
        data = load_yaml(TRAIN_CONFIG)
        del data["domain"]["train"]["friction_coeff"]
        with pytest.raises(ConfigError, match="missing study axes"):
            validate_train_config(data)

    def test_invalid_algo_name_rejected(self):
        data = load_yaml(TRAIN_CONFIG)
        data["algo"]["name"] = "dqn"
        with pytest.raises(ConfigError, match="ppo"):
            validate_train_config(data)

    def test_algo_block_must_exist_for_selected_algo(self):
        data = load_yaml(TRAIN_CONFIG)
        del data["algo"]["sac"]
        data["algo"]["name"] = "sac"
        with pytest.raises(ConfigError, match="no 'algo.sac' block"):
            validate_train_config(data)

    def test_non_monotonic_grid_points_rejected_by_harness(self):
        """Cliff detection compares adjacent points, so a zigzag axis would
        manufacture cliffs out of pure ordering."""
        from eval.ood_harness import OODHarness

        with pytest.raises(ValueError, match="monotonic"):
            OODHarness._assert_monotonic("payload_mass_kg", [1.0, 5.0, 2.0, 9.0])

    def test_monotonic_descending_grid_accepted(self):
        """Friction gets HARDER as it decreases, so descending order is correct."""
        from eval.ood_harness import OODHarness

        OODHarness._assert_monotonic("friction_coeff", [0.9, 0.7, 0.4, 0.3, 0.1])


class TestRewardConfigValidation:
    def test_shipped_reward_config_is_valid(self):
        validate_reward_config(load_yaml(TRAIN_CONFIG)["reward"])

    def test_negative_weight_rejected(self):
        """Penalty signs live in code. A negative config weight would let a YAML
        edit flip a penalty into a bonus."""
        data = load_yaml(TRAIN_CONFIG)
        data["reward"]["terms"]["collision"]["weight"] = -100.0
        with pytest.raises(ConfigError, match="non-negative"):
            validate_reward_config(data["reward"])

    def test_missing_term_field_rejected(self):
        data = load_yaml(TRAIN_CONFIG)
        del data["reward"]["terms"]["collision"]["enabled"]
        with pytest.raises(ConfigError, match="missing 'enabled'"):
            validate_reward_config(data["reward"])

    def test_missing_v1_term_rejected(self):
        data = load_yaml(TRAIN_CONFIG)
        del data["reward"]["terms"]["progress_shaping"]
        with pytest.raises(ConfigError, match="missing required keys"):
            validate_reward_config(data["reward"])

    def test_invalid_version_rejected(self):
        data = load_yaml(TRAIN_CONFIG)
        data["reward"]["version"] = "v3"
        with pytest.raises(ConfigError, match="v1"):
            validate_reward_config(data["reward"])


class TestGravityProhibition:
    def test_gravity_randomization_is_rejected(self):
        """CLAUDE.md principle 7: payload replaces gravity randomization."""
        data = load_yaml(TRAIN_CONFIG)
        data["domain"]["train"]["gravity_range"] = [8.0, 11.0]
        with pytest.raises(ConfigError, match="[Gg]ravity randomization is forbidden"):
            validate_train_config(data)

    def test_nested_gravity_key_also_rejected(self):
        """The check walks the whole tree, so the rule cannot be bypassed by
        hiding the key somewhere unexpected."""
        data = load_yaml(TRAIN_CONFIG)
        data["sim"]["physx"]["randomize_gravity"] = True
        with pytest.raises(ConfigError, match="[Gg]ravity"):
            validate_train_config(data)

    def test_shipped_config_has_no_gravity_randomization(self):
        validate_train_config(load_yaml(TRAIN_CONFIG))


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------
class TestConfigObject:
    def test_dotted_access(self):
        cfg = Config({"a": {"b": {"c": 42}}})
        assert cfg.a.b.c == 42

    def test_missing_key_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="nope"):
            _ = Config({"a": 1}).nope

    def test_get_path_returns_default(self):
        cfg = Config({"a": {"b": 1}})
        assert cfg.get_path("a.b") == 1
        assert cfg.get_path("a.x.y", "fallback") == "fallback"

    def test_to_dict_is_a_deep_copy(self):
        """Mutating the exported dict must not corrupt the loaded config --
        train.py mutates its copy to apply --set overrides."""
        cfg = Config({"a": {"b": 1}})
        data = cfg.to_dict()
        data["a"]["b"] = 999
        assert cfg.a.b == 1


# ---------------------------------------------------------------------------
# Cross-config consistency
# ---------------------------------------------------------------------------
class TestCrossConfigConsistency:
    def test_payload_normalizer_exceeds_ood_ceiling(self):
        """If the observation normalizer saturated below the OOD payload ceiling,
        heavy payloads would be indistinguishable to the policy and the payload
        robustness curve would flatten for a representational reason."""
        train_data = load_yaml(TRAIN_CONFIG)
        eval_data = load_yaml(EVAL_CONFIG)
        normalizer = float(train_data["observations"]["payload_mass_norm_kg"])
        ood_max = float(eval_data["domain"]["ood"]["payload_mass_kg"][1])
        assert normalizer >= ood_max

    def test_nav2_limits_match_policy_action_ranges(self):
        """Nav2 and the policy must drive the same robot within the same limits,
        or the head-to-head comparison is not apples-to-apples."""
        train_data = load_yaml(TRAIN_CONFIG)
        nav2_data = load_yaml(REPO_ROOT / "baselines" / "nav2_params.yaml")

        assert float(nav2_data["nav2"]["controller"]["max_lin_vel"]) == pytest.approx(
            max(train_data["action"]["lin_vel_range"])
        )
        assert float(nav2_data["nav2"]["controller"]["max_ang_vel"]) == pytest.approx(
            max(train_data["action"]["ang_vel_range"])
        )
        assert float(nav2_data["nav2"]["controller"]["goal_tolerance_m"]) == pytest.approx(
            train_data["env"]["goal_tolerance_m"]
        )
        assert float(nav2_data["nav2"]["costmap"]["resolution_m"]) == pytest.approx(
            train_data["env"]["grid_resolution_m"]
        )

    def test_obstacle_pool_can_hold_max_density_layout(self):
        """The simulator's fixed obstacle pool must be able to represent the
        densest configured layout, or high-density OOD points get silently
        truncated and the geometry axis flattens for a scene-budget reason."""
        import math

        train_data = load_yaml(TRAIN_CONFIG)
        terrain = train_data["terrain"]
        usable = train_data["env"]["terrain_size_m"] - 2 * terrain["border_margin_m"]
        target_area = terrain["max_obstacle_area_fraction"] * usable**2
        mean_radius = sum(terrain["obstacle_radius_range_m"]) / 2
        needed = target_area / (math.pi * mean_radius**2)
        assert terrain["max_obstacles_per_env"] >= needed
