"""Token usage per agent task, from the task.py ledger or the Codex session logs.

    python scripts/agents/usage.py                     # ledger: <run-dir>/usage.jsonl
    python scripts/agents/usage.py --task xtts-stream  # one task, per step
    python scripts/agents/usage.py --codex-history xtts-stream text-c
        # tasks run before the ledger existed: Codex sessions whose cwd ends in
        # that worktree name (the before/after baseline)

Codex counts are model input tokens (cached included) and output; Claude
counts split cache reads and writes, plus the cost the CLI reports. They are
different meters: compare a task with its own baseline, not Codex with Claude.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task import main_repo  # noqa: E402

CODEX_SESSIONS = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "sessions"


def last_token_count(session_file: Path) -> dict | None:
    """Final cumulative usage of one Codex session log."""
    last = None
    with open(session_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"token_count"' not in line:
                continue
            try:
                p = json.loads(line).get("payload") or {}
            except ValueError:
                continue
            if p.get("type") == "token_count" and p.get("info"):
                last = p["info"].get("total_token_usage")
    return last


def session_cwd(session_file: Path) -> str:
    with open(session_file, encoding="utf-8", errors="replace") as f:
        try:
            first = json.loads(f.readline())
        except ValueError:
            return ""
    return (first.get("payload") or {}).get("cwd", "") if first.get("type") == "session_meta" else ""


def fill_from_session(entry: dict, sessions: Path = CODEX_SESSIONS) -> dict:
    """`codex exec --json` usage missing: read the session log by thread id."""
    if entry.get("harness") != "codex" or entry.get("input_tokens") or not entry.get("thread_id"):
        return entry
    for f in sessions.rglob(f"*{entry['thread_id']}.jsonl"):
        u = last_token_count(f)
        if u:
            entry = {**entry, "input_tokens": u.get("input_tokens", 0),
                     "cached_input_tokens": u.get("cached_input_tokens", 0),
                     "output_tokens": u.get("output_tokens", 0)}
        break
    return entry


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(fill_from_session(json.loads(line)))
        except ValueError:
            pass
    return rows


def summarize(rows: list[dict], by: str) -> list[tuple[str, dict]]:
    agg: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        a = agg[str(r.get(by, "?"))]
        a["runs"] += 1
        a["seconds"] += r.get("seconds") or 0
        if r.get("harness") == "codex":
            a["codex_in"] += r.get("input_tokens") or 0
            a["codex_cached"] += r.get("cached_input_tokens") or 0
            a["codex_out"] += r.get("output_tokens") or 0
        else:
            a["claude_cache_read"] += r.get("cache_read_input_tokens") or 0
            a["claude_cache_write"] += r.get("cache_creation_input_tokens") or 0
            a["claude_out"] += r.get("output_tokens") or 0
            a["claude_usd"] += r.get("cost_usd") or 0
    return sorted(agg.items())


def fmt(rows: list[tuple[str, dict]], label: str) -> str:
    m = lambda v: f"{v / 1e6:.2f}M"  # noqa: E731
    lines = [f"{label:<22} runs  codex_in  (cached)  codex_out  claude_read  claude_write  claude_out  claude_usd"]
    for key, a in rows:
        lines.append(f"{key:<22} {int(a['runs']):>4}  {m(a['codex_in']):>8}  {m(a['codex_cached']):>8}"
                     f"  {int(a['codex_out']):>9}  {m(a['claude_cache_read']):>11}"
                     f"  {m(a['claude_cache_write']):>12}  {int(a['claude_out']):>10}  {a['claude_usd']:>10.2f}")
    return "\n".join(lines)


def codex_history(names: list[str], sessions: Path = CODEX_SESSIONS) -> list[tuple[str, dict]]:
    agg: dict[str, dict] = {n: defaultdict(float) for n in names}
    for f in sessions.rglob("*.jsonl"):
        leaf = session_cwd(f).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if leaf in agg:
            u = last_token_count(f) or {}
            a = agg[leaf]
            a["runs"] += 1
            a["codex_in"] += u.get("input_tokens", 0)
            a["codex_cached"] += u.get("cached_input_tokens", 0)
            a["codex_out"] += u.get("output_tokens", 0)
    return list(agg.items())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", type=Path, help="default: $AGENT_RUN_DIR or <main clone>/../dw-runs")
    ap.add_argument("--task", help="show one task per step")
    ap.add_argument("--codex-history", nargs="+", metavar="WORKTREE",
                    help="sum Codex sessions by worktree folder name (no ledger needed)")
    a = ap.parse_args(argv)
    if a.codex_history:
        print(fmt(codex_history(a.codex_history), "worktree"))
        return 0
    run_dir = a.run_dir or Path(os.environ.get("AGENT_RUN_DIR", main_repo().parent / "dw-runs"))
    rows = read_ledger(run_dir / "usage.jsonl")
    if a.task:
        print(fmt(summarize([r for r in rows if r.get("task") == a.task], "step"), "step"))
    else:
        print(fmt(summarize(rows, "task"), "task"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
