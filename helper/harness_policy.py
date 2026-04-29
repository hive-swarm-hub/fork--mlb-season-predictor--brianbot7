"""Declarative harness policy: which fields to pass to the hosted model.

This module intentionally does *not* learn feature importance at runtime. The
hive agent (Claude/Codex/etc.) should inspect traces and eval losses, then edit
the selected groups or provide a JSON override. Runtime code only applies that
agent-authored decision to build a label-safe Grok payload.

Optional: set ``MLB_HARNESS_POLICY_PATH`` to a JSON file (repo-relative or
absolute) with either:
  - ``team_keys`` / ``player_keys`` for direct field lists, or
  - ``selected_team_groups`` / ``selected_player_groups`` for group names.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TARGET_KEYS: frozenset[str] = frozenset(
    {
        "actual_wins",
        "overall_rank",
        "league_rank",
        "division_rank",
        "made_playoffs",
        "won_division",
        "league_champion",
        "world_series_champion",
    }
)

# Always include for interpretability and league/rank context.
MANDATORY_TEAM_KEYS: tuple[str, ...] = (
    "season",
    "checkpoint",
    "team_id",
    "team_name",
    "league",
    "division",
)

PLAYER_BASE_KEYS: tuple[str, ...] = (
    "player_name",
    "age",
    "position",
    "role",
)

# Candidate groups are the editable harness surface. The hive agent should move
# fields between groups or change the selected group constants after looking at
# traces and loss deltas.
TEAM_FEATURE_GROUPS: dict[str, list[str]] = {
    "projection_war": [
        "projection_blend_war",
        "dc_team_war",
        "steamer_team_war",
        "zips_team_war",
        "pos_war",
        "sp_war",
        "rp_war",
    ],
    "rotation_top7": [f"sp{i}_war" for i in range(1, 8)],
    "pen_depth": [
        "bp_high_lev_war",
        "rotation_depth_war",
    ],
    "defense": [
        "catcher_framing_runs",
        "def_blend_runs",
    ],
    "run_quality": [
        "pythag_win_pct",
        "baseruns_win_pct",
        "third_order_win_pct",
        "prev_win_pct",
    ],
    "schedule_park": [
        "schedule_strength",
        "intradivision_schedule_strength",
        "division_strength",
        "park_factor_3yr",
    ],
    "durability": [
        "injury_war_lost",
        "durability_risk",
        "age_risk",
    ],
    "checkpoint_form": [
        "checkpoint_wins_above_pace",
    ],
}

PLAYER_FEATURE_GROUPS: dict[str, list[str]] = {
    "war_usage": [
        "projection_blend_war",
        "proj_pa",
        "proj_ip",
    ],
    "hitting_run": [
        "woba",
        "xwoba",
        "barrel_pct",
        "hard_hit_pct",
        "exit_velocity",
        "chase_pct",
    ],
    "pitching_stuff": [
        "stuff_plus",
        "location_plus",
        "projected_leverage_index",
    ],
    "defense_value": [
        "framing_runs",
        "drs_runs",
        "uzr_runs",
        "oaa_runs",
        "def_runs",
    ],
    "durability_risk": [
        "durability_score",
        "injury_risk",
    ],
}

# Current agent-authored default. Future hive iterations should edit these
# directly, or point MLB_HARNESS_POLICY_PATH at a JSON override.
DEFAULT_SELECTED_TEAM_GROUPS: tuple[str, ...] = (
    "projection_war",
    "pen_depth",
    "run_quality",
    "schedule_park",
    "checkpoint_form",
    "defense",
    "durability",
)

DEFAULT_SELECTED_PLAYER_GROUPS: tuple[str, ...] = (
    "war_usage",
    "durability_risk",
)

FALLBACK_TEAM_KEYS: list[str] = list(MANDATORY_TEAM_KEYS) + [
    key
    for group in DEFAULT_SELECTED_TEAM_GROUPS
    for key in TEAM_FEATURE_GROUPS.get(group, [])
]

FALLBACK_PLAYER_KEYS: list[str] = list(PLAYER_BASE_KEYS) + [
    key
    for group in DEFAULT_SELECTED_PLAYER_GROUPS
    for key in PLAYER_FEATURE_GROUPS.get(group, [])
]


def _dedupe_preserve(keys: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _drop_forbidden(keys: list[str]) -> list[str]:
    return [key for key in keys if key not in FORBIDDEN_TARGET_KEYS]


def _group_keys(groups: list[str] | tuple[str, ...], registry: dict[str, list[str]]) -> list[str]:
    keys: list[str] = []
    for group in groups:
        keys.extend(registry.get(group, []))
    return _drop_forbidden(_dedupe_preserve(keys))


def _team_keys_from_groups(groups: list[str] | tuple[str, ...]) -> list[str]:
    return _dedupe_preserve([*MANDATORY_TEAM_KEYS, *_group_keys(groups, TEAM_FEATURE_GROUPS)])


def _player_keys_from_groups(groups: list[str] | tuple[str, ...]) -> list[str]:
    return _dedupe_preserve([*PLAYER_BASE_KEYS, *_group_keys(groups, PLAYER_FEATURE_GROUPS)])


def _policy_path() -> Path | None:
    path_str = os.getenv("MLB_HARNESS_POLICY_PATH", "").strip()
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _policy_from_path(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    selected_team_groups = list(data.get("selected_team_groups", []))
    selected_player_groups = list(data.get("selected_player_groups", []))

    if "team_keys" in data:
        team_keys = _drop_forbidden(_dedupe_preserve([*MANDATORY_TEAM_KEYS, *list(data["team_keys"])]))
    else:
        if not selected_team_groups:
            selected_team_groups = list(DEFAULT_SELECTED_TEAM_GROUPS)
        team_keys = _team_keys_from_groups(selected_team_groups)

    if "player_keys" in data:
        player_keys = _drop_forbidden(_dedupe_preserve([*PLAYER_BASE_KEYS, *list(data["player_keys"])]))
    else:
        if not selected_player_groups:
            selected_player_groups = list(DEFAULT_SELECTED_PLAYER_GROUPS)
        player_keys = _player_keys_from_groups(selected_player_groups)

    return {
        "source": "file",
        "team_keys": team_keys,
        "player_keys": [k for k in player_keys if k not in PLAYER_BASE_KEYS],
        "selected_groups": [
            *selected_team_groups,
            *[f"player__{g}" for g in selected_player_groups],
        ],
        "available_team_groups": sorted(TEAM_FEATURE_GROUPS),
        "available_player_groups": sorted(PLAYER_FEATURE_GROUPS),
        "notes": data.get("notes", ""),
    }


_POLICY_CACHE: dict[str, Any] | None = None


def get_harness_policy() -> dict[str, Any]:
    """Return the current harness policy.

    The default policy is deliberately declarative. It does not inspect labels or
    data distributions; optimization comes from the hive agent editing the
    selected groups after studying traces and eval results.
    """
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE

    path = _policy_path()
    if path is not None:
        file_policy = _policy_from_path(path)
        if file_policy is not None:
            _POLICY_CACHE = file_policy
            return _POLICY_CACHE

    _POLICY_CACHE = {
        "source": "default",
        "team_keys": _team_keys_from_groups(DEFAULT_SELECTED_TEAM_GROUPS),
        "player_keys": [
            k
            for k in _player_keys_from_groups(DEFAULT_SELECTED_PLAYER_GROUPS)
            if k not in PLAYER_BASE_KEYS
        ],
        "selected_groups": [
            *DEFAULT_SELECTED_TEAM_GROUPS,
            *[f"player__{g}" for g in DEFAULT_SELECTED_PLAYER_GROUPS],
        ],
        "available_team_groups": sorted(TEAM_FEATURE_GROUPS),
        "available_player_groups": sorted(PLAYER_FEATURE_GROUPS),
        "notes": "Default agent-authored harness policy.",
    }
    return _POLICY_CACHE


def selected_team_keys() -> list[str]:
    return list(get_harness_policy()["team_keys"])


def selected_player_keys() -> list[str]:
    policy = get_harness_policy()
    return _dedupe_preserve([*PLAYER_BASE_KEYS, *policy["player_keys"]])
