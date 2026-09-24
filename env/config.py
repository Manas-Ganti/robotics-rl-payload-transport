"""Config loading, schema validation, and the train/OOD split assertions.

PURE LOGIC -- imports no Isaac Sim. Locally testable (see tests/test_config.py).

The single most important thing in this module is
:func:`assert_no_train_ood_overlap`. The train/OOD split is the entire study:
if training ranges leak into the held-out evaluation ranges, every
generalization number the project produces becomes meaningless. That check runs
automatically on every load of an eval config and raises loudly.

Usage
-----
    from env.config import load_train_config, load_eval_config

    train_cfg = load_train_config("configs/train.yaml")
    eval_cfg  = load_eval_config("configs/eval_ood.yaml")   # asserts the split
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import yaml

# ---------------------------------------------------------------------------
# The five study axes (CLAUDE.md "Independent Variables"). Sensor degradation
# contributes two parameters, so there are six randomizable parameters total.
# This tuple is the single source of truth for which keys get split-checked.
# ---------------------------------------------------------------------------
STUDY_AXES: Tuple[str, ...] = (
    "obstacle_density",
    "slope_angle_deg",
    "friction_coeff",
    "payload_mass_kg",
    "depth_dropout_prob",
    "depth_noise_std",
)

SCHEMA_VERSION = 1


class ConfigError(ValueError):
    """Raised when a config is malformed or violates a study invariant."""


# ---------------------------------------------------------------------------
# Range primitive
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Range:
    """An inclusive closed interval ``[low, high]`` for one study parameter."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not isinstance(self.low, (int, float)) or not isinstance(self.high, (int, float)):
            raise ConfigError(f"Range bounds must be numeric, got ({self.low!r}, {self.high!r})")
        if self.low > self.high:
            raise ConfigError(f"Range low ({self.low}) exceeds high ({self.high})")

    @classmethod
    def from_yaml(cls, value: Any, *, key: str) -> "Range":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise ConfigError(f"Range '{key}' must be a [low, high] pair, got {value!r}")
        return cls(float(value[0]), float(value[1]))

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def overlaps(self, other: "Range") -> bool:
        """True if the two closed intervals share ANY point (touching counts).

        Touching endpoints (e.g. [0, 10] and [10, 20]) are treated as an
        overlap on purpose: a value of exactly 10.0 would then be simultaneously
        in-distribution and held-out, which is precisely the ambiguity the split
        exists to prevent.
        """
        return self.low <= other.high and other.low <= self.high

    def as_tuple(self) -> Tuple[float, float]:
        return (self.low, self.high)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.low}, {self.high}]"


# ---------------------------------------------------------------------------
# Dotted-access config wrapper
# ---------------------------------------------------------------------------
class Config(Mapping[str, Any]):
    """Read-only mapping with dotted access, so ``cfg.env.num_envs`` works.

    Kept deliberately thin: configs stay plain dicts underneath so they can be
    logged to W&B, serialized next to checkpoints, and diffed between runs.
    """

    def __init__(self, data: Mapping[str, Any], *, source: str | None = None) -> None:
        self._data: Dict[str, Any] = dict(data)
        self._source = source

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value, source=self._source) if isinstance(value, dict) else value

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- Dotted access ------------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            src = f" (from {self._source})" if self._source else ""
            raise AttributeError(f"Config has no key '{key}'{src}") from exc

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Fetch ``a.b.c`` without raising on missing intermediate keys."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return Config(node, source=self._source) if isinstance(node, dict) else node

    def to_dict(self) -> Dict[str, Any]:
        """Deep copy as plain dicts -- for W&B logging and checkpoint sidecars."""
        return copy.deepcopy(self._data)

    @property
    def source(self) -> str | None:
        return self._source

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({sorted(self._data)}, source={self._source!r})"


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------
def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a plain dict, with clear errors on failure."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(data).__name__}: {p}")
    return data


def _resolve_relative(path_value: str, *, relative_to: Path) -> Path:
    """Resolve a config-declared path against CWD first, then the config's dir.

    Configs reference each other by repo-relative path (``configs/train.yaml``),
    which works when run from the repo root. Falling back to the referring
    config's directory keeps things working when invoked from elsewhere.
    """
    candidate = Path(path_value)
    if candidate.is_file():
        return candidate
    sibling = relative_to.parent / candidate.name
    if sibling.is_file():
        return sibling
    nested = relative_to.parent.parent / candidate
    if nested.is_file():
        return nested
    raise ConfigError(
        f"Referenced config '{path_value}' not found "
        f"(tried CWD-relative and relative to {relative_to})"
    )


# ---------------------------------------------------------------------------
# Range extraction
# ---------------------------------------------------------------------------
def extract_ranges(block: Mapping[str, Any], *, block_name: str) -> Dict[str, Range]:
    """Turn a ``domain.train`` / ``domain.ood`` block into ``{axis: Range}``.

    Unknown keys are rejected rather than ignored -- a typo'd axis name would
    otherwise silently drop a randomization axis from the entire study.
    """
    unknown = set(block) - set(STUDY_AXES)
    if unknown:
        raise ConfigError(
            f"Unknown study axes in '{block_name}': {sorted(unknown)}. "
            f"Valid axes: {list(STUDY_AXES)}"
        )
    return {key: Range.from_yaml(value, key=f"{block_name}.{key}") for key, value in block.items()}


# ---------------------------------------------------------------------------
# THE SACRED ASSERTIONS
# ---------------------------------------------------------------------------
def assert_no_train_ood_overlap(
    train_ranges: Mapping[str, Range],
    ood_ranges: Mapping[str, Range],
) -> None:
    """Assert train and OOD ranges are disjoint on EVERY shared axis.

    This is CLAUDE.md principle 2 ("the train/OOD split is sacred") made
    mechanical. It runs on every eval config load.

    Raises
    ------
    ConfigError
        If any axis has overlapping (or merely touching) train and OOD ranges.
    """
    violations: list[str] = []
    for axis in sorted(set(train_ranges) & set(ood_ranges)):
        train_r, ood_r = train_ranges[axis], ood_ranges[axis]
        if train_r.overlaps(ood_r):
            violations.append(f"  {axis}: train={train_r} overlaps ood={ood_r}")

    if violations:
        raise ConfigError(
            "TRAIN/OOD SPLIT VIOLATED -- held-out ranges leak into training ranges.\n"
            + "\n".join(violations)
            + "\n\nThe generalization study is only meaningful if these are disjoint.\n"
            "Fix the ranges in configs/train.yaml or configs/eval_ood.yaml.\n"
            "Do NOT relax this assertion."
        )

    missing = sorted(set(train_ranges) - set(ood_ranges))
    if missing:
        # Not fatal: an axis may intentionally have no OOD counterpart yet.
        # Surfaced as a warning so it is a deliberate choice, not an oversight.
        import warnings

        warnings.warn(
            f"Axes present in train ranges but absent from OOD ranges: {missing}. "
            "These axes will not be swept for generalization.",
            stacklevel=2,
        )


def assert_nominals_in_train_range(
    nominal: Mapping[str, float],
    train_ranges: Mapping[str, Range],
) -> None:
    """Assert every nominal (axis-pinning) value is in-distribution.

    When sweeping one axis, all other axes are pinned to their nominal value.
    If a nominal value were itself OOD, degradation could not be attributed to
    the swept axis -- the experiment would be confounded.
    """
    violations: list[str] = []
    for axis, value in nominal.items():
        if axis not in train_ranges:
            raise ConfigError(f"Nominal value given for unknown axis '{axis}'")
        if not train_ranges[axis].contains(float(value)):
            violations.append(f"  {axis}: nominal={value} is outside train range {train_ranges[axis]}")

    if violations:
        raise ConfigError(
            "Nominal values must be IN-DISTRIBUTION so that single-axis sweeps are\n"
            "not confounded by an off-axis distribution shift.\n" + "\n".join(violations)
        )


def classify_point(value: float, train_range: Range, ood_range: Range | None) -> str:
    """Label one grid point as ``"train"``, ``"ood"``, or ``"gap"``.

    ``"gap"`` means the point sits between the two ranges -- neither trained on
    nor part of the declared held-out set. Gap points are legitimate and useful
    (they probe the boundary), but they are labelled distinctly so plots do not
    misreport them as held-out results.
    """
    if train_range.contains(value):
        return "train"
    if ood_range is not None and ood_range.contains(value):
        return "ood"
    return "gap"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
_REQUIRED_TRAIN_KEYS: Tuple[str, ...] = (
    "seed",
    "phases",
    "sim",
    "env",
    "action",
    "observations",
    "domain",
    "terrain",
    "solvability",
    "reward",
    "algo",
    "logging",
)

_REQUIRED_EVAL_KEYS: Tuple[str, ...] = ("seed", "eval", "domain", "nominal", "grid", "metrics")

_REQUIRED_REWARD_TERMS: Tuple[str, ...] = (
    "goal_reached",
    "collision",
    "step_cost",
    "progress_shaping",
)


def _require_keys(data: Mapping[str, Any], required: Iterable[str], *, what: str) -> None:
    missing = [k for k in required if k not in data]
    if missing:
        raise ConfigError(f"{what} is missing required keys: {missing}")


def _check_schema_version(data: Mapping[str, Any], *, what: str) -> None:
    version = data.get("schema_version")
    if version is None:
        raise ConfigError(f"{what} is missing 'schema_version'")
    if version != SCHEMA_VERSION:
        raise ConfigError(
            f"{what} has schema_version={version}, expected {SCHEMA_VERSION}. "
            "Migrate the config or bump SCHEMA_VERSION in env/config.py."
        )


def validate_reward_config(reward: Mapping[str, Any]) -> None:
    """Validate reward term structure: every term needs enabled + weight."""
    if "terms" not in reward:
        raise ConfigError("reward config is missing 'terms'")
    terms = reward["terms"]
    _require_keys(terms, _REQUIRED_REWARD_TERMS, what="reward.terms")

    for name, spec in terms.items():
        if not isinstance(spec, Mapping):
            raise ConfigError(f"reward.terms.{name} must be a mapping, got {type(spec).__name__}")
        for field in ("enabled", "weight"):
            if field not in spec:
                raise ConfigError(f"reward.terms.{name} is missing '{field}'")
        if not isinstance(spec["enabled"], bool):
            raise ConfigError(f"reward.terms.{name}.enabled must be a bool")
        if not isinstance(spec["weight"], (int, float)):
            raise ConfigError(f"reward.terms.{name}.weight must be numeric")
        if spec["weight"] < 0:
            raise ConfigError(
                f"reward.terms.{name}.weight must be non-negative ({spec['weight']} given). "
                "Penalty terms carry their negative sign in env/reward.py, not in config, "
                "so that a weight's sign can never silently flip a term's meaning."
            )

    version = reward.get("version", "v1")
    if version not in ("v1", "v2"):
        raise ConfigError(f"reward.version must be 'v1' or 'v2', got {version!r}")


def validate_train_config(data: Mapping[str, Any]) -> None:
    """Full structural validation of a training config."""
    _check_schema_version(data, what="train config")
    _require_keys(data, _REQUIRED_TRAIN_KEYS, what="train config")

    # Gravity randomization is forbidden (CLAUDE.md principle 7). Checked FIRST:
    # a gravity key in domain.train would otherwise be rejected by the generic
    # unknown-axis check, and the error would not say WHY it is not allowed.
    _assert_no_gravity_randomization(data)

    if "train" not in data["domain"]:
        raise ConfigError("train config is missing 'domain.train' ranges")
    train_ranges = extract_ranges(data["domain"]["train"], block_name="domain.train")

    missing_axes = sorted(set(STUDY_AXES) - set(train_ranges))
    if missing_axes:
        raise ConfigError(f"domain.train is missing study axes: {missing_axes}")

    validate_reward_config(data["reward"])

    if data["env"]["num_envs"] < 1:
        raise ConfigError("env.num_envs must be >= 1")
    if data["env"]["episode_length_s"] <= 0:
        raise ConfigError("env.episode_length_s must be > 0")
    if data["sim"]["dt"] <= 0:
        raise ConfigError("sim.dt must be > 0")
    if data["sim"]["decimation"] < 1:
        raise ConfigError("sim.decimation must be >= 1")

    algo_name = data["algo"]["name"]
    if algo_name not in ("ppo", "sac"):
        raise ConfigError(f"algo.name must be 'ppo' or 'sac', got {algo_name!r}")
    if algo_name not in data["algo"]:
        raise ConfigError(f"algo.name is '{algo_name}' but no 'algo.{algo_name}' block exists")

    for name in ("lin_vel_range", "ang_vel_range"):
        Range.from_yaml(data["action"][name], key=f"action.{name}")

    if data["solvability"]["connectivity"] not in (4, 8):
        raise ConfigError("solvability.connectivity must be 4 or 8")
    if data["solvability"]["robot_radius_m"] <= 0:
        raise ConfigError("solvability.robot_radius_m must be > 0")
    if data["env"]["grid_resolution_m"] <= 0:
        raise ConfigError("env.grid_resolution_m must be > 0")


def _assert_no_gravity_randomization(data: Mapping[str, Any]) -> None:
    """Reject any gravity-randomization key anywhere in the config tree.

    CLAUDE.md principle 7: payload mass is the deployment-realistic dynamics
    axis; gravity is not. This walks the whole tree so the rule cannot be
    bypassed by nesting the key somewhere unexpected.
    """
    offenders: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_l = str(key).lower()
                if "gravity" in key_l and ("random" in key_l or "range" in key_l):
                    offenders.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(data, "config")
    if offenders:
        raise ConfigError(
            "Gravity randomization is forbidden in this project (CLAUDE.md principle 7): "
            f"payload mass is the dynamics axis, not gravity. Offending keys: {offenders}"
        )


def validate_eval_config(data: Mapping[str, Any]) -> None:
    """Full structural validation of an eval config (before cross-checks)."""
    _check_schema_version(data, what="eval config")
    _require_keys(data, _REQUIRED_EVAL_KEYS, what="eval config")

    if "ood" not in data["domain"]:
        raise ConfigError("eval config is missing 'domain.ood' ranges")
    extract_ranges(data["domain"]["ood"], block_name="domain.ood")

    if data["eval"]["episodes_per_cell"] < 1:
        raise ConfigError("eval.episodes_per_cell must be >= 1")

    for axis, spec in data["grid"].items():
        if axis not in STUDY_AXES:
            raise ConfigError(f"grid contains unknown axis '{axis}'. Valid: {list(STUDY_AXES)}")
        if "points" not in spec:
            raise ConfigError(f"grid.{axis} is missing 'points'")
        if not spec["points"]:
            raise ConfigError(f"grid.{axis}.points is empty")
        if "enabled" not in spec:
            raise ConfigError(f"grid.{axis} is missing 'enabled'")


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------
def load_train_config(path: str | Path = "configs/train.yaml") -> Config:
    """Load + validate a training config. Merges in the referenced robot config."""
    path = Path(path)
    data = load_yaml(path)
    validate_train_config(data)

    robot_path_value = data.get("robot_config")
    if robot_path_value:
        robot_path = _resolve_relative(str(robot_path_value), relative_to=path)
        robot_data = load_yaml(robot_path)
        _check_schema_version(robot_data, what=f"robot config ({robot_path})")
        for key in ("robot", "payload", "sensors"):
            if key not in robot_data:
                raise ConfigError(f"robot config is missing '{key}': {robot_path}")
        data = {**data, **{k: v for k, v in robot_data.items() if k != "schema_version"}}

    return Config(data, source=str(path))


def load_eval_config(path: str | Path = "configs/eval_ood.yaml") -> Config:
    """Load + validate an eval config AND enforce the train/OOD split.

    The returned config carries the resolved training config under the ``train``
    key, so downstream code has env/robot settings and both range blocks in one
    object without re-reading files.
    """
    path = Path(path)
    data = load_yaml(path)
    validate_eval_config(data)

    train_path = _resolve_relative(str(data.get("train_config", "configs/train.yaml")), relative_to=path)
    train_cfg = load_train_config(train_path)

    train_ranges = extract_ranges(train_cfg.to_dict()["domain"]["train"], block_name="domain.train")
    ood_ranges = extract_ranges(data["domain"]["ood"], block_name="domain.ood")

    # ---- THE SACRED CHECKS -------------------------------------------------
    assert_no_train_ood_overlap(train_ranges, ood_ranges)
    assert_nominals_in_train_range(data["nominal"], train_ranges)

    merged = dict(data)
    merged["train"] = train_cfg.to_dict()
    return Config(merged, source=str(path))


def get_train_ranges(cfg: Config) -> Dict[str, Range]:
    """Extract ``{axis: Range}`` training ranges from a train OR eval config."""
    data = cfg.to_dict()
    block = data["train"]["domain"]["train"] if "train" in data else data["domain"]["train"]
    return extract_ranges(block, block_name="domain.train")


def get_ood_ranges(cfg: Config) -> Dict[str, Range]:
    """Extract ``{axis: Range}`` held-out ranges from an eval config."""
    data = cfg.to_dict()
    if "ood" not in data.get("domain", {}):
        raise ConfigError("Config has no 'domain.ood' block -- is this an eval config?")
    return extract_ranges(data["domain"]["ood"], block_name="domain.ood")
