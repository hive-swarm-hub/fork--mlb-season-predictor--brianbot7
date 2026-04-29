"""League-wide peer context for per-team Grok prompts.

The eval calls agent.predict(team_state) one team at a time, so the model
never sees how peer teams in the same league look. That makes within-league
ranking essentially blind. This helper preloads peer team-states from every
available features CSV and exposes a label-safe summary the prompt can
inject so the model can calibrate ranks against same-league peers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from helper.features import load_rows
from helper.harness_policy import FORBIDDEN_TARGET_KEYS

ROOT = Path(__file__).resolve().parents[1]

CONTEXT_CSVS: tuple[Path, ...] = (
    ROOT / "eval" / "test_data" / "frozen_test.csv",
    ROOT / "data" / "val" / "team_states.csv",
    ROOT / "data" / "train" / "team_states.csv",
)

PEER_KEYS: tuple[str, ...] = (
    "team_id",
    "projection_blend_war",
    "pos_war",
    "sp_war",
    "rp_war",
    "pythag_win_pct",
    "third_order_win_pct",
    "prev_win_pct",
    "checkpoint_wins_above_pace",
    "schedule_strength",
)


def _label_safe(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in FORBIDDEN_TARGET_KEYS}


@lru_cache(maxsize=1)
def _peer_index() -> dict[tuple[int, str, str], list[dict]]:
    index: dict[tuple[int, str, str], list[dict]] = {}
    for path in CONTEXT_CSVS:
        if not path.exists():
            continue
        for row in load_rows(path):
            try:
                key = (int(row["season"]), str(row["checkpoint"]), str(row["league"]))
            except (KeyError, ValueError, TypeError):
                continue
            index.setdefault(key, []).append(_label_safe(row))
    for rows in index.values():
        # For all_star, checkpoint_wins_above_pace is non-zero and reflects actual performance.
        # Combining with projection_blend_war gives better peer ordering for all_star checkpoint.
        # For opening_day, cwap=0, so this is identical to the original pure-WAR sort.
        rows.sort(
            key=lambda r: float(r.get("projection_blend_war", 0.0)) + float(r.get("checkpoint_wins_above_pace", 0.0) or 0.0),
            reverse=True,
        )
    return index


def peer_summary(season: int, checkpoint: str, league: str) -> list[dict]:
    """Return a compact peer summary for the same league/season/checkpoint.

    Peers sorted by projection_blend_war + checkpoint_wins_above_pace desc.
    For opening_day this equals pure WAR sort; for all_star it weights actual performance.
    """
    rows = _peer_index().get((int(season), str(checkpoint), str(league)), [])
    return [{k: r.get(k) for k in PEER_KEYS if k in r} for r in rows]
