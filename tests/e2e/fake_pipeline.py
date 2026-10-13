"""Run ``system.main`` with the scripted fakes. Invoked in a subprocess by tests/e2e.

    python -m tests.e2e.fake_pipeline <hydra overrides...>

Writes the captured LLM calls to ``$E2E_CAPTURE`` (JSON) when set.
"""

from __future__ import annotations

import json
import os
import sys

from tests.e2e import fakes


def _patch() -> None:
    import env_setup.build_llms as build_llms
    import env_setup.sandbox_utils as sandbox_utils
    import system.main as main_mod
    import system.research_one_step as research_one_step

    def build_llm_and_embedder(*args, **kwargs):
        return fakes.ScriptedChatModel(backend="primary"), fakes.fake_embedder()

    def build_llm_pool(endpoints_cfg, keys=None):
        wanted = list(endpoints_cfg) if keys is None else [k for k in keys if k in endpoints_cfg]
        pool = {str(k): fakes.ScriptedChatModel(backend=str(k)) for k in wanted}
        pool["primary"] = fakes.ScriptedChatModel(backend="primary")
        return pool

    for mod in (main_mod, build_llms):
        mod.build_llm_and_embedder = build_llm_and_embedder
        mod.build_llm_pool = build_llm_pool
    for mod in (main_mod, research_one_step, sandbox_utils):
        mod.SandboxClient = fakes.FakeSandboxClient


def main() -> None:
    _patch()
    import system.main as main_mod

    args = sys.argv[1:]
    if "--config-path" not in args:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args = [
            "--config-path",
            os.path.join(repo, "sys_config"),
            "--config-name",
            "ascollab",
            *args,
        ]
    sys.argv = ["system.main", *args]
    try:
        main_mod.main()
    finally:
        out = os.environ.get("E2E_CAPTURE")
        if out:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(fakes.CAPTURE, fh, indent=1)


if __name__ == "__main__":
    main()
