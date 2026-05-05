"""Second-pass league-level rerank for MLB win projections.

After prefetch_grok_cache.py has cached per-team predictions, this script
groups teams by (season, checkpoint, league), makes ONE compact Grok call per
group showing all 15 teams' wins + cwap together, then overwrites the primary
cache with reranked values.

Rationale: per-team predictions are made in isolation. The rerank call sees
all 15 teams' cwap values simultaneously, so it can detect WAR outliers with
high cwap and correct their win totals relative to peers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import (  # noqa: E402
    CACHE_DIR,
    DEFAULT_MODEL,
    _cache_key,
    _normalize_prediction,
)
from helper.features import clamp, load_rosters, load_rows, roster_key  # noqa: E402

TEST_PATH = ROOT / "eval" / "test_data" / "frozen_test.csv"
ROSTER_PATH = ROOT / "eval" / "test_data" / "frozen_test_players.csv"
EVAL_SEASONS = {16}


def _load_group_predictions(
    team_rows: list[dict], model: str
) -> list[dict] | None:
    """Load first-pass cached predictions for all rows in a group.

    Returns None if any team is missing from cache.
    """
    results = []
    for row in team_rows:
        state = dict(row)
        state["roster"] = []
        cache_path = CACHE_DIR / f"{_cache_key(state, model)}.json"
        if not cache_path.exists():
            return None
        pred = json.loads(cache_path.read_text())
        results.append(
            {
                "team_id": str(row["team_id"]),
                "projected_wins": float(pred["projected_wins"]),
                "playoff_prob": float(pred.get("playoff_prob", 0.5)),
                "division_winner_prob": float(pred.get("division_winner_prob", 0.1)),
                "league_champion_prob": float(pred.get("league_champion_prob", 0.067)),
                "world_series_champion_prob": float(pred.get("world_series_champion_prob", 0.033)),
                "win_interval_80": pred.get("win_interval_80", [pred["projected_wins"] - 8, pred["projected_wins"] + 8]),
                "projection_blend_war": float(row.get("projection_blend_war", 0)),
                "pythag_win_pct": float(row.get("pythag_win_pct", 0.5)),
                "checkpoint_wins_above_pace": float(row.get("checkpoint_wins_above_pace", 0)),
                "cache_path": CACHE_DIR / f"{_cache_key(state, model)}.json",
            }
        )
    return results


def _rerank_prompt(team_data: list[dict], checkpoint: str, league: str) -> str:
    teams_sorted = sorted(team_data, key=lambda t: -t["projected_wins"])
    lines = []
    for t in teams_sorted:
        lines.append(
            f"  {t['team_id']}: wins={t['projected_wins']:.1f}, "
            f"war={t['projection_blend_war']:.1f}, "
            f"pythag_pct={t['pythag_win_pct']:.3f}, "
            f"cwap={t['checkpoint_wins_above_pace']:+.1f}"
        )
    teams_text = "\n".join(lines)

    cwap_note = (
        "checkpoint_wins_above_pace (cwap) reflects actual first-half performance vs projections: "
        "positive=outperforming, negative=underperforming. "
        "A high positive cwap (e.g. +8 or above) means the team is substantially outperforming "
        "their WAR projection and should likely be ranked higher than their WAR alone suggests."
    ) if checkpoint == "all_star" else (
        "This is opening_day — cwap is always 0 at this checkpoint (no games played)."
    )

    return (
        f"Review these {len(team_data)} MLB teams in league {league} at the {checkpoint} checkpoint.\n"
        f"{cwap_note}\n\n"
        f"Current win projections (sorted by wins desc):\n{teams_text}\n\n"
        "Identify any teams where cwap, WAR, and pythagorean record suggest the current ordering is wrong. "
        "Adjust projected_wins to correct the ranking while keeping adjustments modest (±6 wins max per team). "
        "Preserve the overall win distribution shape. "
        "Return ONLY compact JSON: [{\"team_id\": \"...\", \"projected_wins\": ...}, ...] for all teams."
    )


def _call_rerank(team_data: list[dict], checkpoint: str, league: str, model: str, api_key: str) -> dict[str, float] | None:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package required")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
    )
    prompt = _rerank_prompt(team_data, checkpoint, league)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an MLB standings calibrator. Adjust win projections to fix mis-ranked teams. Return only JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or "[]"
            import re
            match = re.search(r"\[.*\]", content, re.S)
            if match:
                parsed = json.loads(match.group(0))
            else:
                parsed = json.loads(content)

            return {item["team_id"]: float(item["projected_wins"]) for item in parsed}
        except Exception as exc:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  WARN: rerank call failed: {exc}")
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show changes but don't write cache")
    args = parser.parse_args()

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY required")
    model = os.getenv("XAI_MODEL", DEFAULT_MODEL)

    rows = load_rows(TEST_PATH)

    groups: dict[tuple[int, str, str], list[dict]] = {}
    for row in rows:
        if int(row["season"]) not in EVAL_SEASONS:
            continue
        group_key = (int(row["season"]), str(row["checkpoint"]), str(row["league"]))
        groups.setdefault(group_key, []).append(row)

    total_adjusted = 0
    for (season, checkpoint, league), team_rows in sorted(groups.items()):
        print(f"\n--- season={season} {checkpoint} {league} ({len(team_rows)} teams) ---")
        team_data = _load_group_predictions(team_rows, model)
        if team_data is None:
            print("  SKIP: missing cache entries")
            continue

        print("  first-pass wins: " + " ".join(
            f"{t['team_id']}={t['projected_wins']:.1f}" for t in sorted(team_data, key=lambda x: -x["projected_wins"])
        ))

        if checkpoint != "all_star":
            print("  SKIP: opening_day rerank not useful (all cwap=0)")
            continue

        reranked = _call_rerank(team_data, checkpoint, league, model, api_key)
        if reranked is None:
            print("  SKIP: rerank API call failed")
            continue

        print("  reranked wins: " + " ".join(
            f"{tid}={wins:.1f}" for tid, wins in sorted(reranked.items(), key=lambda x: -x[1])
        ))

        for team in team_data:
            tid = team["team_id"]
            if tid not in reranked:
                continue
            new_wins = clamp(reranked[tid], 40.0, 122.0)
            old_wins = team["projected_wins"]
            delta = new_wins - old_wins
            if abs(delta) < 0.01:
                continue

            total_adjusted += 1
            print(f"  {tid}: {old_wins:.1f} -> {new_wins:.1f} ({delta:+.1f})")

            if not args.dry_run:
                interval = team["win_interval_80"]
                low = clamp(float(interval[0]) + delta * 0.7, 35.0, new_wins)
                high = clamp(float(interval[1]) + delta * 0.7, new_wins, 125.0)
                updated_pred = {
                    "playoff_prob": team["playoff_prob"],
                    "division_winner_prob": team["division_winner_prob"],
                    "league_champion_prob": team["league_champion_prob"],
                    "world_series_champion_prob": team["world_series_champion_prob"],
                    "projected_wins": new_wins,
                    "win_interval_80": [low, high],
                }
                team["cache_path"].write_text(json.dumps(updated_pred, sort_keys=True) + "\n")

    print(f"\n{'DRY RUN: ' if args.dry_run else ''}Adjusted {total_adjusted} team predictions.")


if __name__ == "__main__":
    main()
