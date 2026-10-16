"""Run one task end to end: Codex implements, Claude Code reviews, Codex fixes.

    python scripts/agents/task.py <name> <branch> --prompt P.md --title T.txt \\
        --body B.md --commit-msg M.txt [--reviewer claude|codex] [--max-rounds 3]

The loop runs here, not in the tech lead's session: implement -> full suite ->
commit/push/PR -> @rev + @sec review -> fix -> ... until both reviews have no
high/medium finding or the round limit is hit. The operator gets a few status
lines and one RESULT line; prompts, outputs and full reviews stay in the run
folder and on the PR.

Reviewers default to Claude Code headless (`claude -p --agent marc:review` and
`marc:security`, read-only tools), which keeps the Codex budget for writing
code and keeps the review independent of the vendor that wrote it.

Exit codes: 0 both reviews clean, 1 findings left after --max-rounds,
2 Codex usage limit, 3 the full suite still fails after the fix attempts,
4 a reviewer failed or returned no verdict line (needs the operator),
5 Codex left nothing to commit or review.

Every Codex and Claude run appends its token usage to <run-dir>/usage.jsonl;
`python scripts/agents/usage.py` summarizes it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
EXIT_OK, EXIT_FINDINGS, EXIT_LIMIT, EXIT_TESTS, EXIT_REVIEWER, EXIT_NO_CHANGES = 0, 1, 2, 3, 4, 5


def main_repo() -> Path:
    """The main clone, also when this script runs from one of its worktrees."""
    r = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                       cwd=HERE, capture_output=True, text=True)
    return Path(r.stdout.strip()).parent if r.returncode == 0 and r.stdout.strip() else HERE

# Appended to every Codex prompt. Each tool round trip resends the whole
# context, so polling a running command and dumping big files are what burn
# the budget (measured on the XTTS stream task: ~20% of it was test polling).
CODEX_RULES = """
## Working efficiently (token budget is tight)
- Run tests with ONE exec_command and a long wait (yield_time_ms >= 300000). Do not poll a running command every second.
- Focused tests only: `{focused}`. Never run the whole suite; the caller does. If pytest hits a temp-dir permission error, use another folder under .test-tmp; do not investigate ACLs.
- Read narrowly: `rg -n` for symbols, then only the needed line ranges (<= ~120 lines at a time). Do not dump whole files or long test modules.
- Keep command output short (tail/filters, -q, max_output_tokens). Do not re-read files you just edited.
"""

REVIEW_FOCUS = {
    "rev": "correctness and quality: bugs, regressions, race conditions, behavior "
           "changes, missing or weak tests, maintainability, consistency with the "
           "surrounding code and with AGENTS.md",
    "sec": "security: command/argument injection, unsafe subprocess or path handling "
           "(e.g. executables resolved from cwd/PATH), untrusted file parsing, secrets "
           "or private data in logs, resource exhaustion/DoS, unsafe deserialization, "
           "privilege issues",
}
CLAUDE_AGENT = {"rev": "marc:review", "sec": "marc:security"}
# Read-only: the reviewer inspects the diff; this script posts its report.
CLAUDE_REVIEW_ALLOWED = [
    "Read", "Grep", "Glob",
    "Bash(git diff:*)", "Bash(git show:*)", "Bash(git log:*)",
    "Bash(git status:*)", "Bash(git rev-parse:*)", "Bash(git merge-base:*)",
    "Bash(rg:*)",
]
CLAUDE_REVIEW_DENIED = ["Edit", "Write", "NotebookEdit", "Bash(gh:*)", "Bash(git push:*)",
                        "Bash(git commit:*)"]

VERDICT_RE = re.compile(
    r"^\s*@(rev|sec) verdict:\s*(PASS|ADVISE|BLOCK)\s*;\s*high=(\d+)\s+medium=(\d+)\s+low=(\d+)\s*$",
    re.IGNORECASE | re.MULTILINE)
LIMIT_RE = re.compile(r"try again at ([^.\"\n]*)", re.IGNORECASE)

DEFAULT_ADD_PATHS = ["debora_whisper", "tests", "docs", "scripts", "README.md", "AGENTS.md",
                     "pyproject.toml", "uv.lock", "requirements-test.txt", ".github"]


class CodexLimit(Exception):
    pass


@dataclass
class Verdict:
    kind: str
    verdict: str = "NONE"
    high: int = 0
    medium: int = 0
    low: int = 0
    ok: bool = False          # a verdict line was found
    report: str = ""
    error: str = ""

    @property
    def clean(self) -> bool:
        return self.ok and self.high == 0 and self.medium == 0

    def short(self) -> str:
        if not self.ok:
            return f"{self.kind}=NO-VERDICT({self.error or 'missing line'})"
        return f"{self.kind}={self.verdict}(h{self.high}/m{self.medium}/l{self.low})"


def parse_verdict(kind: str, text: str) -> Verdict:
    """The last `@<kind> verdict: ...; high=N medium=N low=N` line wins."""
    v = Verdict(kind=kind, report=text)
    for m in VERDICT_RE.finditer(text or ""):
        if m.group(1).lower() == kind:
            v.verdict = m.group(2).upper()
            v.high, v.medium, v.low = (int(m.group(i)) for i in (3, 4, 5))
            v.ok = True
    return v


def codex_limit_message(text: str) -> str | None:
    m = LIMIT_RE.search(text or "")
    return m.group(0) if m else None


def codex_usage(events_text: str) -> dict:
    """Sum `turn.completed` usage from `codex exec --json` output."""
    total = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "turns": 0}
    thread = None
    for line in (events_text or "").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "thread.started":
            thread = ev.get("thread_id")
        if ev.get("type") == "turn.completed":
            u = ev.get("usage") or {}
            total["turns"] += 1
            for k in ("input_tokens", "cached_input_tokens", "output_tokens"):
                total[k] += int(u.get(k) or 0)
    total["thread_id"] = thread
    return total


def claude_usage(result: dict) -> dict:
    u = result.get("usage") or {}
    return {
        "input_tokens": u.get("input_tokens", 0),
        "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "turns": result.get("num_turns"),
        "cost_usd": result.get("total_cost_usd"),
        "session_id": result.get("session_id"),
    }


def default_python(repo: Path) -> str:
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        p = repo / rel
        if p.exists():
            return str(p)
    return sys.executable


def which(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"{name} not found on PATH")
    return path


def lower_priority_kwargs() -> dict:
    """Child processes (and theirs) inherit the priority: the full suite
    otherwise makes typing lag across Windows."""
    if os.name == "nt":
        return {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    return {"preexec_fn": lambda: os.nice(10)}


@dataclass
class Task:
    name: str
    branch: str
    prompt: Path | None
    title: str
    body: Path | None
    commit_msg: Path | None
    worktree: Path
    run_dir: Path
    repo: Path
    python: str
    reviewer: str = "claude"
    max_rounds: int = 3
    test_fix_attempts: int = 2
    add_paths: list[str] = field(default_factory=lambda: list(DEFAULT_ADD_PATHS))
    out = print  # replaced in tests

    # ---- helpers -------------------------------------------------------
    def say(self, msg: str) -> None:
        self.out(f"[{self.name}] {msg}", flush=True)

    def file(self, suffix: str) -> Path:
        return self.run_dir / f"{self.name}-{suffix}"

    def record(self, entry: dict) -> None:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": self.name, **entry}
        with open(self.run_dir / "usage.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def git(self, *args: str, cwd: Path | None = None, check: bool = True) -> str:
        r = subprocess.run(["git", *args], cwd=cwd or self.worktree, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        if check and r.returncode:
            raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()[-500:]}")
        return r.stdout.strip()

    def gh(self, *args: str, check: bool = True) -> str:
        r = subprocess.run(["gh", *args], cwd=self.worktree, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if check and r.returncode:
            raise SystemExit(f"gh {' '.join(args[:3])} failed: {r.stderr.strip()[-500:]}")
        return r.stdout.strip()

    # ---- steps ---------------------------------------------------------
    def ensure_worktree(self) -> None:
        self.git("fetch", "-q", "origin", cwd=self.repo)
        if not self.worktree.exists():
            self.git("worktree", "add", "-q", "-b", self.branch, str(self.worktree),
                     "origin/main", cwd=self.repo)
            self.say(f"worktree {self.worktree} on {self.branch} from origin/main")

    def focused_cmd(self) -> str:
        return f"{self.python} -m pytest -q -p no:cacheprovider <files> --basetemp .test-tmp/pt"

    def codex(self, effort: str, prompt_text: str, step: str) -> str:
        """Run Codex in the worktree; return its final message."""
        prompt_file = self.file(f"{step}.prompt")
        prompt_file.write_text(prompt_text + "\n" + CODEX_RULES.format(focused=self.focused_cmd()),
                               encoding="utf-8")
        out, log = self.file(f"{step}.out"), self.file(f"{step}.log")
        started = time.monotonic()
        with open(prompt_file, encoding="utf-8") as stdin, open(log, "w", encoding="utf-8") as lf:
            r = subprocess.run([which("codex"), "exec", "--json", "-c",
                                f"model_reasoning_effort={effort}", "--sandbox", "workspace-write",
                                "-C", str(self.worktree), "-o", str(out), "-"],
                               stdin=stdin, stdout=lf, stderr=subprocess.STDOUT)
        text = log.read_text(encoding="utf-8", errors="replace")
        self.record({"step": step, "harness": "codex", "effort": effort,
                     "seconds": round(time.monotonic() - started), "exit": r.returncode,
                     **codex_usage(text)})
        limit = codex_limit_message(text)
        if limit:
            raise CodexLimit(limit)
        return out.read_text(encoding="utf-8", errors="replace") if out.exists() else ""

    def clean_scratch(self) -> None:
        for pattern in (".tmp", ".pytest-*", ".test-tmp*", ".tmp-*", "pytest-cache-files-*"):
            for p in self.worktree.glob(pattern):
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)

    def run_suite(self) -> tuple[bool, str]:
        """Each tests/test_*.py in its own low-priority process: one process for
        the whole suite grows to many GB. Returns (passed, summary)."""
        base = self.run_dir / f"{self.name}-suite"
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True)
        passed = failed = 0
        failures: list[str] = []
        for test in sorted((self.worktree / "tests").glob("test_*.py")):
            r = subprocess.run([self.python, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                f"tests/{test.name}", "--basetemp", str(base / test.stem)],
                               cwd=self.worktree, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", **lower_priority_kwargs())
            text = r.stdout + r.stderr
            m = re.search(r"(\d+) passed", text)
            passed += int(m.group(1)) if m else 0
            mf = re.search(r"(\d+) (failed|errors?)", text)
            if mf:
                failed += int(mf.group(1))
            elif not m and r.returncode not in (0, 5):  # 5 = no tests collected
                failed += 1
                failures.append(f"ERROR tests/{test.name} (no summary)")
            failures += [l for l in text.splitlines() if l.startswith(("FAILED ", "ERROR "))]
        summary = f"{failed} failed, {passed} passed" if failed else f"{passed} passed"
        self.file("tests.txt").write_text("\n".join(failures + [summary]) + "\n", encoding="utf-8")
        return failed == 0, summary

    def commit_and_push(self, message: str) -> None:
        self.git("add", "-A", "--", *[p for p in self.add_paths if (self.worktree / p).exists()])
        left = self.git("status", "--porcelain")
        if left:
            self.say(f"not committed (outside add paths): {' '.join(l[3:] for l in left.splitlines())[:300]}")
        msg_file = self.file("commit.txt")
        msg_file.write_text(message, encoding="utf-8")
        self.git("commit", "-q", "-F", str(msg_file))
        self.git("push", "-q", "-u", "origin", self.branch)

    def ensure_pr(self) -> str:
        pr = self.gh("pr", "view", self.branch, "--json", "number", "-q", ".number", check=False)
        if not pr:
            args = ["pr", "create", "--base", "main", "--head", self.branch, "--title", self.title]
            args += ["--body-file", str(self.body)] if self.body else ["--body", ""]
            self.gh(*args)
            pr = self.gh("pr", "view", self.branch, "--json", "number", "-q", ".number")
        return pr

    def review_prompt(self, kind: str, sha: str) -> str:
        marker = f"## @{kind} review"
        return (
            f"REPORT-ONLY, READ-ONLY review. Do not post comments, do not modify files, "
            f"do not run gh. The caller posts your report on the PR.\n\n"
            f"Review ONLY the diff `git diff origin/main...HEAD` of the branch checked out "
            f"in this worktree (HEAD {sha}); read surrounding code as needed.\n"
            f"Focus: {REVIEW_FOCUS[kind]}.\n"
            f"Report each finding with file:line, severity (high/medium/low), the concrete "
            f"failure scenario and a suggested fix. Ignore pure style nits.\n"
            f"Start the report with `{marker}` and the reviewed SHA. End with exactly one final "
            f"line: `@{kind} verdict: <PASS|ADVISE|BLOCK>; high=<n> medium=<n> low=<n>`."
        )

    def review(self, kind: str, sha: str) -> Verdict:
        prompt = self.review_prompt(kind, sha)
        step = f"review-{kind}"
        self.file(f"{step}.prompt").write_text(prompt, encoding="utf-8")
        started = time.monotonic()
        if self.reviewer == "claude":
            cmd = [which("claude"), "-p", "--agent", CLAUDE_AGENT[kind], "--output-format", "json",
                   "--allowedTools", *CLAUDE_REVIEW_ALLOWED,
                   "--disallowedTools", *CLAUDE_REVIEW_DENIED]
            r = subprocess.run(cmd, input=prompt, cwd=self.worktree, capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
            self.file(f"{step}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
            try:
                result = json.loads(r.stdout)
            except ValueError:
                return Verdict(kind=kind, error=f"claude exit {r.returncode}, no JSON")
            self.record({"step": step, "harness": "claude", "agent": CLAUDE_AGENT[kind],
                         "seconds": round(time.monotonic() - started), "exit": r.returncode,
                         **claude_usage(result)})
            text = result.get("result") or ""
            if result.get("is_error"):
                v = Verdict(kind=kind, error=f"claude error: {text[:120]}")
                v.report = text
                return v
        else:
            out, log = self.file(f"{step}.out"), self.file(f"{step}.log")
            with open(log, "w", encoding="utf-8") as lf:
                r = subprocess.run([which("codex"), "exec", "--json", "-c",
                                    "model_reasoning_effort=medium", "--sandbox", "read-only",
                                    "-C", str(self.worktree), "-o", str(out), "-"],
                                   input=prompt, stdout=lf, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8")
            logtext = log.read_text(encoding="utf-8", errors="replace")
            self.record({"step": step, "harness": "codex", "effort": "medium",
                         "seconds": round(time.monotonic() - started), "exit": r.returncode,
                         **codex_usage(logtext)})
            limit = codex_limit_message(logtext)
            if limit:
                raise CodexLimit(limit)
            text = out.read_text(encoding="utf-8", errors="replace") if out.exists() else ""
        self.file(f"{step}.md").write_text(text, encoding="utf-8")
        return parse_verdict(kind, text)

    def post_review(self, pr: str, v: Verdict, sha: str) -> None:
        body = v.report.strip() or f"(empty report: {v.error})"
        marker = f"## @{v.kind} review"
        if not body.startswith(marker):
            who = "Claude Code headless" if self.reviewer == "claude" else "Codex"
            body = f"{marker} ({who}, {sha})\n\n{body}"
        f = self.file(f"review-{v.kind}-comment.md")
        f.write_text(body + "\n", encoding="utf-8")
        self.gh("pr", "comment", pr, "--body-file", str(f))

    def reviews(self, pr: str, sha: str) -> list[Verdict]:
        results: dict[str, Verdict] = {}

        def run(kind: str) -> None:
            try:
                results[kind] = self.review(kind, sha)
            except CodexLimit as e:
                results[kind] = Verdict(kind=kind, error=f"CODEX LIMIT {e}")
            except Exception as e:  # reported, never treated as a pass
                results[kind] = Verdict(kind=kind, error=f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=run, args=(k,)) for k in ("rev", "sec")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        verdicts = [results["rev"], results["sec"]]
        for v in verdicts:
            if v.report or v.ok:
                self.post_review(pr, v, sha)
        return verdicts

    def fix_prompt(self, findings: str) -> str:
        return (f"Continue this work (branch {self.branch}, already committed). "
                f"The reviews requested changes:\n\n{findings}\n\n"
                f"Fix every high/medium finding with regression tests. Do NOT commit. "
                f"End with a 3-line summary.")

    def fix_commit_message(self, summary: str) -> str:
        return f"fix: address the review\n\n{summary.strip()[:1200]}\n"

    # ---- the loop ------------------------------------------------------
    def run(self, fix_file: Path | None = None, skip_impl: bool = False) -> int:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        pr = sha = "-"
        verdicts: list[Verdict] = []
        try:
            self.ensure_worktree()
            fix_summary = ""
            if fix_file:
                fix_summary = self.codex("medium", self.fix_prompt(fix_file.read_text(encoding="utf-8")),
                                         "fix-manual")
                self.say("codex fix done")
            elif not skip_impl and not self.git("log", "--oneline", "origin/main..HEAD"):
                if not self.prompt:
                    raise SystemExit("--prompt is required for the first implementation")
                self.codex("high", self.prompt.read_text(encoding="utf-8") + "\n\nDo NOT commit.",
                           "impl")
                self.say("codex implementation done")
            first_commit = not fix_file
            reviewed_sha = None
            for round_no in range(self.max_rounds + 1):
                self.clean_scratch()
                if self.git("status", "--porcelain"):
                    for attempt in range(self.test_fix_attempts + 1):
                        ok, summary = self.run_suite()
                        self.say(f"suite: {summary}")
                        if ok:
                            break
                        if attempt == self.test_fix_attempts:
                            return self.finish(EXIT_TESTS, pr, sha, verdicts)
                        failures = self.file("tests.txt").read_text(encoding="utf-8")
                        self.codex("medium", "The full suite failed:\n\n" + failures[-4000:] +
                                   "\n\nFix the failures (do NOT commit).", f"fixtests-{round_no}-{attempt}")
                        self.clean_scratch()
                    if first_commit:
                        if not self.commit_msg:
                            raise SystemExit("--commit-msg is required for the first commit")
                        message = self.commit_msg.read_text(encoding="utf-8")
                        first_commit = False
                    else:
                        message = self.fix_commit_message(fix_summary)
                    self.commit_and_push(message)
                if not self.git("log", "--oneline", "origin/main..HEAD"):
                    self.say("no commits ahead of origin/main")
                    return self.finish(EXIT_NO_CHANGES, pr, sha, verdicts)
                sha = self.git("rev-parse", "--short", "HEAD")
                if sha == reviewed_sha:
                    self.say("codex fix changed nothing; not re-reviewing the same commit")
                    return self.finish(EXIT_FINDINGS, pr, sha, verdicts)
                reviewed_sha = sha
                pr = self.ensure_pr()
                self.say(f"PR #{pr} at {sha}; reviewing ({self.reviewer})")
                verdicts = self.reviews(pr, sha)
                self.say(" ".join(v.short() for v in verdicts))
                if any(not v.ok for v in verdicts):
                    if any("CODEX LIMIT" in v.error for v in verdicts):
                        return self.finish(EXIT_LIMIT, pr, sha, verdicts)
                    return self.finish(EXIT_REVIEWER, pr, sha, verdicts)
                if all(v.clean for v in verdicts):
                    return self.finish(EXIT_OK, pr, sha, verdicts)
                if round_no == self.max_rounds:
                    break
                findings = "\n\n".join(v.report for v in verdicts if not v.clean)
                fix_summary = self.codex("medium", self.fix_prompt(findings), f"fix-{round_no + 1}")
                self.say(f"codex fix round {round_no + 1} done")
            return self.finish(EXIT_FINDINGS, pr, sha, verdicts)
        except CodexLimit as e:
            self.say(f"CODEX LIMIT ({e})")
            return self.finish(EXIT_LIMIT, pr, sha, verdicts)

    def finish(self, code: int, pr: str, sha: str, verdicts: list[Verdict]) -> int:
        reviews = " ".join(v.short() for v in verdicts) or "reviews=none"
        self.out(f"RESULT {self.name} exit={code} pr=#{pr} sha={sha} {reviews} "
                 f"run_dir={self.run_dir}", flush=True)
        return code


def read_text_arg(value: str | None) -> str:
    if value and Path(value).is_file():
        return Path(value).read_text(encoding="utf-8").strip()
    return value or ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name", help="short task name; also the worktree folder name")
    ap.add_argument("branch")
    ap.add_argument("--prompt", type=Path, help="Codex task prompt (first implementation)")
    ap.add_argument("--title", default="", help="PR title, or a file holding it")
    ap.add_argument("--body", type=Path, help="PR body file")
    ap.add_argument("--commit-msg", type=Path, help="message file for the first commit")
    ap.add_argument("--reviewer", choices=("claude", "codex"), default="claude")
    ap.add_argument("--max-rounds", type=int, default=3, help="review/fix rounds after the first")
    ap.add_argument("--fix", type=Path, help="findings file: run one Codex fix round first")
    ap.add_argument("--skip-impl", action="store_true", help="worktree already has the changes")
    ap.add_argument("--worktree-root", type=Path,
                    default=None, help="default: $AGENT_WORKTREE_ROOT or <main clone>/../dw-wt")
    ap.add_argument("--run-dir", type=Path,
                    default=None, help="default: $AGENT_RUN_DIR or <main clone>/../dw-runs")
    ap.add_argument("--python", help="interpreter for the test suite (default: main clone .venv)")
    a = ap.parse_args(argv)
    repo = main_repo()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", a.name):
        ap.error("name must be a plain folder name")
    wt_root = a.worktree_root or Path(os.environ.get("AGENT_WORKTREE_ROOT", repo.parent / "dw-wt"))
    run_dir = a.run_dir or Path(os.environ.get("AGENT_RUN_DIR", repo.parent / "dw-runs"))
    task = Task(name=a.name, branch=a.branch, prompt=a.prompt, title=read_text_arg(a.title),
                body=a.body, commit_msg=a.commit_msg, worktree=wt_root / a.name, run_dir=run_dir,
                repo=repo, python=a.python or default_python(repo), reviewer=a.reviewer,
                max_rounds=a.max_rounds)
    return task.run(fix_file=a.fix, skip_impl=a.skip_impl)


if __name__ == "__main__":
    sys.exit(main())
