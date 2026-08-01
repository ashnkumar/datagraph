import pytest

from datagraph.cli import main


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
