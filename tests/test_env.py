import os

import pytest

from datagraph.env import load_env_file


@pytest.fixture
def env_file(tmp_path):
    def _write(text: str):
        path = tmp_path / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    return _write


def test_a_missing_file_is_not_an_error(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == {}


def test_pairs_reach_the_environment(env_file, monkeypatch):
    monkeypatch.delenv("DG_TEST_KEY", raising=False)

    loaded = load_env_file(env_file("DG_TEST_KEY=abc123\n"))

    assert loaded == {"DG_TEST_KEY": "abc123"}
    assert os.environ["DG_TEST_KEY"] == "abc123"


def test_the_real_environment_wins(env_file, monkeypatch):
    monkeypatch.setenv("DG_TEST_KEY", "exported")

    loaded = load_env_file(env_file("DG_TEST_KEY=from-file\n"))

    # An explicit export beats a file the user forgot they wrote.
    assert loaded == {}
    assert os.environ["DG_TEST_KEY"] == "exported"


def test_comments_blanks_and_junk_are_skipped(env_file, monkeypatch):
    monkeypatch.delenv("DG_TEST_KEY", raising=False)

    loaded = load_env_file(env_file("# a comment\n\nnot an assignment\n  DG_TEST_KEY = spaced \n"))

    assert loaded == {"DG_TEST_KEY": "spaced"}


def test_quotes_and_export_prefixes_are_stripped(env_file, monkeypatch):
    monkeypatch.delenv("DG_TEST_QUOTED", raising=False)
    monkeypatch.delenv("DG_TEST_EXPORTED", raising=False)

    loaded = load_env_file(env_file('DG_TEST_QUOTED="quoted"\nexport DG_TEST_EXPORTED=exported\n'))

    assert loaded == {"DG_TEST_QUOTED": "quoted", "DG_TEST_EXPORTED": "exported"}


def test_an_unfilled_placeholder_does_not_shadow_a_real_value(env_file, monkeypatch):
    # This is the shape `.env.example` ships with: the key is named but left blank. Setting
    # it to "" would make the SDK see a key that is present and empty.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")

    load_env_file(env_file("ANTHROPIC_API_KEY=\n"))

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-real"


def test_a_blank_value_is_never_set_at_all(env_file, monkeypatch):
    monkeypatch.delenv("DG_TEST_BLANK", raising=False)

    assert load_env_file(env_file("DG_TEST_BLANK=\n")) == {}
    assert "DG_TEST_BLANK" not in os.environ
