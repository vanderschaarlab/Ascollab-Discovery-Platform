"""Module loggers must not be shadowed inside functions.

In a module that defines ``log = logging.getLogger(__name__)``, a function that also binds ``log``
makes every closure inside it see the local value instead of the logger. That once passed a
stats object as ``logger=`` to the content-filter retry, which then crashed on ``.warning``.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("agents", "env_setup", "ranker", "sandbox", "scripts", "stores", "system")


def _binds_log(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "log" and isinstance(node.ctx, ast.Store)
    return isinstance(node, ast.arg) and node.arg == "log"


def test_module_logger_is_not_rebound_in_functions():
    offenders = set()
    for package in PACKAGES:
        for path in sorted((REPO_ROOT / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            defines_log = any(
                isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "log" for t in stmt.targets)
                for stmt in tree.body
            )
            if not defines_log:
                continue
            for fn in ast.walk(tree):
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    offenders.update(
                        f"{path.relative_to(REPO_ROOT)}:{n.lineno}"
                        for n in ast.walk(fn)
                        if _binds_log(n)
                    )

    assert sorted(offenders) == []
