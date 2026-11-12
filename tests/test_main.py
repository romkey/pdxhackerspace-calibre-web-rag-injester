from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from typing import Any

import pytest

from calibre_web2rag import __main__ as cli
from calibre_web2rag import __version__
from calibre_web2rag.config import Settings

SettingsFactory = Callable[..., Settings]  # the make_settings fixture


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    # Never read a developer's real .env during tests.
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


def _use_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"calibre-web2rag {__version__}\n"


def test_bad_config_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CALIBRE_METADATA_DB", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "CALIBRE_METADATA_DB" in capsys.readouterr().err


def test_success_returns_0(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _use_settings(monkeypatch, make_settings())
    monkeypatch.setattr(cli, "ingest", lambda settings, stop: 42)
    with caplog.at_level("INFO"):
        assert cli.main([]) == 0
    assert "Ingest complete. Upserted 42 chunks" in caplog.text


def test_failure_returns_1(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _use_settings(monkeypatch, make_settings())

    def _boom(settings: Settings, stop: threading.Event) -> int:
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(cli, "ingest", _boom)
    assert cli.main([]) == 1
    assert "Ingestion failed" in caplog.text
    assert "qdrant unreachable" in caplog.text


def test_sigterm_stops_ingest(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _use_settings(monkeypatch, make_settings())
    # Any: signal handlers take (signum, frame).
    handlers: dict[int, Callable[..., Any]] = {}
    monkeypatch.setattr(signal, "signal", handlers.__setitem__)

    def _ingest(settings: Settings, stop: threading.Event) -> int:
        # SIGTERM arrives mid-run: the handler sets the stop event.
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert stop.is_set()
        return 7

    monkeypatch.setattr(cli, "ingest", _ingest)
    with caplog.at_level("INFO"):
        assert cli.main([]) == 0
    assert "Stopped early. Upserted 7 chunks" in caplog.text
