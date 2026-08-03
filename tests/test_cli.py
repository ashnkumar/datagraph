import pytest
from rich.console import Console

from datagraph.cli import ConfigError, _model, main
from datagraph.env import load_env_file


@pytest.fixture
def run(capsys):
    def _run(*argv):
        code = main(list(argv))
        return code, capsys.readouterr().out

    return _run


def test_demo_runs_and_reports_a_settled_query(run):
    code, out = run("demo")

    assert code == 0
    assert "Answer" in out
    assert "efficient" in out
    assert "Ledger invariants hold" in out


def test_demo_never_prints_a_suppressed_value(run):
    _, out = run("demo")
    assert "synthetic-participant" not in out
    assert "1992-02-29" not in out


def test_demo_accepts_an_engine_choice(run):
    code, out = run("demo", "--engine", "leave_one_out")
    assert code == 0
    assert "leave_one_out" in out


def test_compare_shows_every_engine(run):
    code, out = run("compare")

    assert code == 0
    for engine in ("shapley", "exact_shapley", "leave_one_out"):
        assert engine in out


def test_providers_lists_the_seeded_registry(run):
    code, out = run("providers")

    assert code == 0
    assert "Aurora Sleep Cohort" in out
    assert "Delta Metrics Group" in out


def test_a_query_matching_nothing_reports_a_refund(run):
    code, out = run("demo", "--question", "zzz nothing matches this")

    assert code == 0
    assert "Refunded" in out


def test_errors_are_reported_without_a_traceback(run):
    # Escrowing more than the researcher holds is a ledger error, not a crash.
    code, out = run("demo", "--payment", "999999999")

    assert code == 1
    assert "LedgerError" in out
    assert "Traceback" not in out


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit) as excinfo:
        main(["nonsense"])
    assert excinfo.value.code != 0


def test_live_without_a_key_explains_what_to_do(run, monkeypatch, tmp_path):
    # Somewhere with no .env, and with nothing exported, so this is a genuinely unconfigured
    # machine rather than whatever the developer happens to have set.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    code, out = run("demo", "--live")

    assert code == 2
    assert "ANTHROPIC_API_KEY" in out
    assert ".env" in out
    # The SDK's own error names a class and an internal concept; this one names the fix.
    assert "TypeError" not in out
    assert "Traceback" not in out


def test_a_key_in_a_dotenv_file_satisfies_the_live_check(monkeypatch, tmp_path):
    """The README tells the reader to put their key in `.env`; this is that promise."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-the-file\n", encoding="utf-8")

    load_env_file()

    # Constructing the client is not a network call, so this reaches the real code path
    # without spending anything.
    assert _model(live=True, console=Console()) is not None


def test_the_live_check_is_what_rejects_an_unconfigured_machine(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ConfigError):
        _model(live=True, console=Console())
