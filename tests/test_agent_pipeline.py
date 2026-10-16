"""scripts/agents: the Codex -> Claude review loop and its usage ledger.

No real codex, claude or gh runs here: the loop is driven through a fake Task.
"""
import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

AGENTS = Path(__file__).resolve().parent.parent / "scripts" / "agents"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"agents_{name}", AGENTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up
    sys.path.insert(0, str(AGENTS))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(AGENTS))
    return mod


task = _load("task")
usage = _load("usage")


# ---- parsing ---------------------------------------------------------------

def test_verdict_last_matching_line_wins():
    text = ("## @rev review\n@rev verdict: PASS; high=0 medium=0 low=0\n"
            "@sec verdict: BLOCK; high=2 medium=0 low=0\n"
            "@REV Verdict: advise; high=0 medium=1 low=3\n")
    v = task.parse_verdict("rev", text)
    assert (v.ok, v.verdict, v.high, v.medium, v.low) == (True, "ADVISE", 0, 1, 3)
    assert not v.clean


def test_missing_verdict_is_never_clean():
    v = task.parse_verdict("sec", "## @sec review\nLooks fine to me.")
    assert not v.ok and not v.clean


def test_block_is_never_clean():
    # @rev has a "critical" severity the counters have no slot for
    assert not task.parse_verdict("rev", "@rev verdict: BLOCK; high=0 medium=0 low=0").clean


def test_verdict_line_may_be_quoted_or_bold():
    assert task.parse_verdict("rev", "`@rev verdict: PASS; high=0 medium=0 low=2`").clean
    assert task.parse_verdict("sec", "**@sec verdict: PASS; high=0 medium=0 low=0**").clean


def test_low_only_findings_are_clean():
    assert task.parse_verdict("sec", "@sec verdict: ADVISE; high=0 medium=0 low=4").clean


def test_codex_limit_detected_in_json_events():
    err = ('{"type":"turn.failed","error":{"message":"You\\u2019ve hit your usage limit. '
           'Upgrade to Pro or try again at 1:59 AM."}}')
    assert task.codex_limit_message(err) == "try again at 1:59 AM"
    assert task.codex_limit_message('{"type":"turn.completed"}') is None


def test_limit_words_in_command_output_are_not_a_limit():
    item = {"type": "item.completed", "item": {"type": "command_execution",
            "aggregated_output": 'LIMIT_RE = re.compile(r"try again at ...")'}}
    assert task.codex_limit_message(json.dumps(item)) is None
    assert task.codex_limit_message(json.dumps({"type": "error", "message": "try again at 3 PM."}))


def test_codex_usage_sums_turns():
    events = "\n".join([
        '{"type":"thread.started","thread_id":"abc"}',
        '{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":80,"output_tokens":5}}',
        "not json",
        '{"type":"turn.completed","usage":{"input_tokens":50,"cached_input_tokens":40,"output_tokens":7}}',
    ])
    assert task.codex_usage(events) == {"input_tokens": 150, "cached_input_tokens": 120,
                                        "output_tokens": 12, "turns": 2, "thread_id": "abc"}


# ---- reviewer command ------------------------------------------------------

def _task(tmp_path, cls=None, **kw):
    cls = cls or task.Task
    t = cls(name="demo", branch="feat/demo", prompt=None, title="t", body=None, commit_msg=None,
            worktree=tmp_path / "wt", run_dir=tmp_path / "runs", repo=tmp_path, python=sys.executable,
            **kw)
    t.run_dir.mkdir(parents=True, exist_ok=True)
    t.lines = []
    t.out = lambda msg, flush=False: t.lines.append(msg)
    return t


def test_claude_reviewer_is_headless_and_read_only(tmp_path, monkeypatch):
    calls = []

    class Done:
        returncode = 0
        stderr = ""
        stdout = json.dumps({"result": "## @sec review\n@sec verdict: PASS; high=0 medium=0 low=0",
                             "usage": {"cache_read_input_tokens": 10, "output_tokens": 3},
                             "num_turns": 2, "total_cost_usd": 0.01, "is_error": False})

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return Done()

    monkeypatch.setattr(task.subprocess, "run", fake_run)
    monkeypatch.setattr(task, "which", lambda name: name)
    t = _task(tmp_path)
    v = t.review("sec", "abc123")
    cmd, kw = calls[0]
    assert v.clean and v.verdict == "PASS"
    assert cmd[:4] == ["claude", "-p", "--agent", "marc:security"]
    denied = cmd[cmd.index("--disallowedTools") + 1:]
    # no allow rules: reads outside the working folders are denied in -p mode
    assert "--allowedTools" not in cmd
    assert {"Bash", "Edit", "Write"} <= set(denied)  # no shell: git/rg flags can write or exec
    assert cmd[cmd.index("--permission-mode") + 1] == "default"
    assert cmd[cmd.index("--setting-sources") + 1] == "user"  # ignore worktree .claude/
    assert kw["cwd"] == t.review_dir
    assert "--dangerously-skip-permissions" not in cmd
    assert "abc123" in kw["input"] and "abc123" not in " ".join(cmd)  # prompt on stdin
    ledger = [json.loads(l) for l in (t.run_dir / "usage.jsonl").read_text().splitlines()]
    assert ledger[0]["harness"] == "claude" and ledger[0]["cache_read_input_tokens"] == 10


# ---- the loop --------------------------------------------------------------

class FakeTask(task.Task):
    """Scripted codex/reviews; git state advances one commit per change."""

    def setup(self, verdict_rounds, fix_changes=True, impl_limit=False):
        self.verdict_rounds = list(verdict_rounds)
        self.fix_changes = fix_changes
        self.impl_limit = impl_limit
        self.codex_steps, self.reviewed = [], []
        self.commits, self.changed, self.pushes = 0, False, 0
        return self

    def ensure_worktree(self):
        pass

    def codex(self, effort, prompt_text, step):
        self.codex_steps.append((step, effort))
        if step == "impl" and self.impl_limit:
            raise task.CodexLimit("try again at 1:59 AM")
        if step == "impl" or self.fix_changes:
            self.changed = True
        return "summary of " + step

    def dirty(self):
        return self.changed

    def git(self, *args, cwd=None, check=True):
        if args[0] == "log":
            return "c1 x" if self.commits else ""
        if args[0] == "rev-parse":
            return f"sha{self.commits}"
        return ""

    def run_suite(self):
        return True, "10 passed"

    def commit(self, message):
        self.commits += 1
        self.changed = False

    def push(self):
        self.pushes += 1

    def ensure_pr(self):
        return "77"

    def reviews(self, pr, sha):
        self.reviewed.append(sha)
        texts = self.verdict_rounds.pop(0)
        return [task.parse_verdict(k, texts[k]) for k in ("rev", "sec")]


PASS = {"rev": "@rev verdict: PASS; high=0 medium=0 low=0",
        "sec": "@sec verdict: PASS; high=0 medium=0 low=1"}
BLOCK = {"rev": "x\n@rev verdict: BLOCK; high=1 medium=0 low=0",
         "sec": "@sec verdict: PASS; high=0 medium=0 low=0"}


def _run(tmp_path, *rounds, **setup):
    t = _task(tmp_path, cls=FakeTask).setup(rounds, **setup)
    t.prompt = tmp_path / "p.md"
    t.prompt.write_text("do it")
    t.commit_msg = tmp_path / "m.txt"
    t.commit_msg.write_text("feat: demo")
    return t, t.run()


def test_fix_round_then_clean_reviews(tmp_path):
    t, code = _run(tmp_path, BLOCK, PASS)
    assert code == task.EXIT_OK
    assert t.codex_steps == [("impl", "high"), ("fix-1", "medium")]
    assert t.reviewed == ["sha1", "sha2"]
    assert t.lines[-1].startswith("RESULT demo exit=0 pr=#77 sha=sha2 rev=PASS")


def test_fix_without_changes_is_not_reviewed_again(tmp_path):
    t, code = _run(tmp_path, BLOCK, PASS, fix_changes=False)
    assert code == task.EXIT_FINDINGS
    assert t.reviewed == ["sha1"]


def test_round_limit(tmp_path):
    t = _task(tmp_path, cls=FakeTask, max_rounds=1).setup([BLOCK, BLOCK])
    t.prompt, t.commit_msg = tmp_path / "p.md", tmp_path / "m.txt"
    t.prompt.write_text("x")
    t.commit_msg.write_text("y")
    assert t.run() == task.EXIT_FINDINGS
    assert len(t.reviewed) == 2


def test_reviewer_without_verdict_stops_for_the_operator(tmp_path):
    t, code = _run(tmp_path, {"rev": "no verdict here", "sec": PASS["sec"]})
    assert code == task.EXIT_REVIEWER
    assert "rev=NO-VERDICT" in t.lines[-1]


def test_codex_limit_exits_2(tmp_path):
    t, code = _run(tmp_path, PASS, impl_limit=True)
    assert code == task.EXIT_LIMIT
    assert t.reviewed == []


def test_git_failure_reports_exit_6_with_result_line(tmp_path):
    class PushFails(FakeTask):
        def push(self):
            raise task.TaskError("git push failed: rejected")

    t = _task(tmp_path, cls=PushFails).setup([PASS])
    t.prompt, t.commit_msg = tmp_path / "p.md", tmp_path / "m.txt"
    t.prompt.write_text("x")
    t.commit_msg.write_text("y")
    assert t.run() == task.EXIT_ERROR
    assert t.lines[-1].startswith("RESULT demo exit=6") and "push failed" in t.lines[-1]


def test_a_crashing_review_thread_is_a_reviewer_failure(tmp_path, monkeypatch):
    t = _task(tmp_path)
    monkeypatch.setattr(t, "write_diff", lambda: None)
    monkeypatch.setattr(t, "checkout_for_review", lambda sha: None)
    monkeypatch.setattr(t, "drop_review_checkout", lambda: None)
    monkeypatch.setattr(t, "review_cmd", lambda kind: [])
    monkeypatch.setattr(t, "post_review", lambda *a: None)

    def review(kind, sha):
        if kind == "rev":
            raise SystemExit("claude not found")  # BaseException: threading drops it
        return task.parse_verdict("sec", PASS["sec"])

    monkeypatch.setattr(t, "review", review)
    rev, sec = t.reviews("1", "abc")
    assert not rev.ok and "claude not found" in rev.error and sec.clean


def test_codex_crash_is_an_error_not_no_changes(tmp_path, monkeypatch):
    class Crashed:
        returncode = 1

    def fake_run(cmd, stdout=None, **kw):
        stdout.write('{"type":"error","message":"stream disconnected"}\n')
        return Crashed()

    monkeypatch.setattr(task.subprocess, "run", fake_run)
    monkeypatch.setattr(task, "which", lambda name: name)
    t = _task(tmp_path)
    with pytest.raises(task.TaskError, match="exit 1"):
        t.codex("low", "x", "impl")


@pytest.mark.parametrize("argv", [["demo", "--force"], ["demo", "a/../b"], ["demo"], ["a b", "x"]])
def test_bad_arguments_exit_6(argv):
    assert task.main(argv) == task.EXIT_ERROR


def test_leftovers_outside_add_paths_are_not_dirty(tmp_path):
    def git(*args):
        task.subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("")
    git("add", "tests")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    t = _task(tmp_path)
    t.worktree = tmp_path
    (tmp_path / "notes.md").write_text("scratch")
    assert not t.dirty()
    (tmp_path / "tests" / "test_a.py").write_text("x = 1\n")
    assert t.dirty()


def test_unexpected_error_still_prints_result(tmp_path):
    t = _task(tmp_path, cls=FakeTask).setup([PASS])
    t.prompt = tmp_path / "missing.md"  # mistyped --prompt
    assert t.run() == task.EXIT_ERROR
    assert t.lines[-1].startswith("RESULT demo exit=6") and "FileNotFoundError" in t.lines[-1]


def test_codex_reviewer_crash_never_reuses_an_old_report(tmp_path, monkeypatch):
    class Crashed:
        returncode = 1

    monkeypatch.setattr(task.subprocess, "run", lambda cmd, **kw: Crashed())
    monkeypatch.setattr(task, "which", lambda name: name)
    t = _task(tmp_path, reviewer="codex")
    t.file("review-rev.out").write_text(PASS["rev"])  # previous round
    v = t.review("rev", "abc")
    assert not v.ok and "codex exit 1" in v.error


def test_which_ignores_the_current_folder(tmp_path, monkeypatch):
    for fake in ("git.bat", "git.cmd", "git.exe", "git"):
        (tmp_path / fake).write_text("")
    (tmp_path / "git").chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", os.pathsep.join([".", "", os.environ["PATH"]]))
    assert Path(task.which("git")).parent != tmp_path


def test_secret_pattern_needs_a_whole_token():
    assert not task.SECRET_RE.search("see the task-runner-retry-on-timeout branch")
    assert not task.SECRET_RE.search("ask-for-a-very-long-hyphenated-name-here")
    assert task.SECRET_RE.search("key sk-ant-api03-" + "Ab1" * 10)
    assert task.SECRET_RE.search("token=ghp_" + "a1" * 18)


def test_unpushed_commits_are_pushed_before_review(tmp_path):
    t = _task(tmp_path, cls=FakeTask).setup([PASS])
    t.commits = 1  # rerun after a failed push: ahead, clean tree
    assert t.run() == task.EXIT_OK
    assert t.pushes == 1 and t.codex_steps == []


def test_report_quoting_a_token_is_not_posted(tmp_path, monkeypatch):
    t = _task(tmp_path)
    posted = []
    for name in ("write_diff", "drop_review_checkout"):
        monkeypatch.setattr(t, name, lambda: None)
    monkeypatch.setattr(t, "checkout_for_review", lambda sha: None)
    monkeypatch.setattr(t, "review_cmd", lambda kind: [])
    monkeypatch.setattr(t, "post_review", lambda pr, v, sha: posted.append(v.kind))
    leak = "found it: ghp_" + "a1" * 18 + "\n"
    monkeypatch.setattr(t, "review", lambda kind, sha: task.parse_verdict(
        kind, (leak if kind == "sec" else "") + PASS[kind]))
    rev, sec = t.reviews("1", "abc")
    assert posted == ["rev"] and not sec.ok and "credential" in sec.error


# ---- suite runner ----------------------------------------------------------

def test_run_suite_counts_per_file(tmp_path):
    t = _task(tmp_path)
    tests = t.worktree / "tests"
    tests.mkdir(parents=True)
    (tests / "test_ok.py").write_text("def test_a():\n    pass\n\ndef test_b():\n    pass\n")
    (tests / "test_bad.py").write_text("def test_c():\n    assert False\n")
    ok, summary = t.run_suite()
    assert not ok and summary == "1 failed, 2 passed"
    assert "FAILED tests/test_bad.py::test_c" in (t.run_dir / "demo-tests.txt").read_text()


# ---- usage -----------------------------------------------------------------

def _session(path, cwd, totals):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "session_meta", "payload": {"cwd": cwd}}]
    for inp in totals:
        lines.append({"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": inp, "cached_input_tokens": inp // 2,
                                  "output_tokens": 1}}}})
    path.write_text("\n".join(json.dumps(l) for l in lines))


def test_codex_history_matches_windows_worktree_names(tmp_path):
    _session(tmp_path / "2026/10/16/rollout-a.jsonl", "D:\\repos\\dw-wt\\xtts-stream", [10, 300])
    _session(tmp_path / "2026/10/16/rollout-b.jsonl", "D:\\repos\\dw-wt\\xtts-stream", [50])
    _session(tmp_path / "2026/10/16/rollout-c.jsonl", "D:\\repos\\dw-wt\\other", [999])
    rows = dict(usage.codex_history(["xtts-stream"], sessions=tmp_path))
    assert rows["xtts-stream"]["runs"] == 2 and rows["xtts-stream"]["codex_in"] == 350


def test_ledger_entry_without_usage_reads_the_session(tmp_path):
    _session(tmp_path / "2026/10/16/rollout-x-tid42.jsonl", "D:\\wt", [7, 70])
    entry = {"harness": "codex", "thread_id": "tid42", "input_tokens": 0}
    assert usage.fill_from_session(entry, sessions=tmp_path)["input_tokens"] == 70


def test_summary_by_task(tmp_path):
    rows = [{"task": "a", "harness": "codex", "input_tokens": 1_000_000, "cached_input_tokens": 0,
             "output_tokens": 10},
            {"task": "a", "harness": "claude", "cache_read_input_tokens": 2_000_000,
             "output_tokens": 5, "cost_usd": 0.5}]
    out = usage.fmt(usage.summarize(rows, "task"), "task")
    assert textwrap.dedent(out).splitlines()[1].split()[:3] == ["a", "2", "1.00M"]
