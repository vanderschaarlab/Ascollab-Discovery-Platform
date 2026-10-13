"""run_review_replay runs outside Hydra's @main, so it must set up logging itself."""

import logging

import pytest

from scripts.analysis.reviewer_matching import run_review_replay


class _StopAfterLoggingSetup(Exception):
    pass


def test_entry_point_shows_info_lines_from_pipeline_code(monkeypatch, capsys):
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    saved_lib_levels = {name: logging.getLogger(name).level for name in ("openai", "httpx")}
    root.handlers = []  # as in a plain `python -m` run (pytest adds its own handlers)
    root.setLevel(logging.WARNING)

    def stop() -> None:
        raise _StopAfterLoggingSetup

    monkeypatch.setattr(run_review_replay, "parse_args", stop)
    try:
        with pytest.raises(_StopAfterLoggingSetup):
            run_review_replay.main()
        handlers = root.handlers[:]
        root_level = root.level
        logging.getLogger("system.reviewer_matching").info(
            "[reviewer_matching] sourcing: tool=x tool_share=0.5"
        )
        lib_levels = {name: logging.getLogger(name).level for name in saved_lib_levels}
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        for name, level in saved_lib_levels.items():
            logging.getLogger(name).setLevel(level)

    assert root_level == logging.INFO
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)
    assert "[reviewer_matching] sourcing: tool=x tool_share=0.5" in capsys.readouterr().err
    assert lib_levels == {"openai": logging.WARNING, "httpx": logging.WARNING}
