"""Prefetch Grok prediction cache for the API-first task.

This is a convenience smoke-test helper. It does not change scoring; it simply
calls agent.predict for each frozen feature row so the normal eval can reuse
cached API responses.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helper.features import load_rosters, load_rows, roster_key  # noqa: E402


def load_agent():
    spec = importlib.util.spec_from_file_location("agent", ROOT / "agent.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    agent = load_agent()
    rows = load_rows(ROOT / "eval" / "test_data" / "frozen_test.csv")
    rosters = load_rosters(ROOT / "eval" / "test_data" / "frozen_test_players.csv")
    team_states = []
    for row in rows:
        state = dict(row)
        state["roster"] = rosters.get(roster_key(row), [])
        team_states.append(state)

    def run_one(state: dict) -> tuple[int, str, str, float]:
        pred = agent.predict(state)
        return (
            int(state["season"]),
            str(state["checkpoint"]),
            str(state["team_id"]),
            float(pred["projected_wins"]),
        )

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, state): state for state in team_states}
        for future in as_completed(futures):
            state = futures[future]
            try:
                season, checkpoint, team_id, wins = future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(futures):
                    print(
                        f"cached {completed}/{len(futures)} latest={season} "
                        f"{checkpoint} {team_id} wins={wins:.1f}",
                        flush=True,
                    )
            except Exception as exc:
                failed += 1
                print(
                    f"WARN failed {state.get('team_id')} s{state.get('season')} "
                    f"{state.get('checkpoint')}: {type(exc).__name__}: {exc}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
