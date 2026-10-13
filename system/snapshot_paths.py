import os
from pathlib import Path
from typing import Any


def resolve_project_root(project_dir: str | Path | None = None) -> Path:
    """Resolve project root consistently across Hydra and non-Hydra execution."""
    if project_dir is not None:
        return Path(project_dir).expanduser().resolve()

    # Under Hydra runs, runtime.cwd points to the original project root.
    try:
        from hydra.core.hydra_config import HydraConfig  # type: ignore

        if HydraConfig.initialized():
            runtime_cwd = getattr(HydraConfig.get().runtime, "cwd", None)
            if runtime_cwd:
                return Path(runtime_cwd).expanduser().resolve()
    except Exception:
        pass

    env_project_dir = os.getenv("PROJECT_DIR")
    if env_project_dir:
        return Path(env_project_dir).expanduser().resolve()

    return Path().resolve()


def resolve_snapshot_run_name(
    *,
    run_name: str | None = None,
    exp_config: Any | None = None,
) -> str:
    """Resolve canonical snapshot run name."""
    return (
        run_name
        or getattr(exp_config, "run_name", None)
        or getattr(exp_config, "exp_name", None)
        or "unnamed_experiment"
    )


def resolve_run_snapshot_dir(
    *,
    run_name: str | None = None,
    exp_config: Any | None = None,
    base_snapshots_dir: str = "snapshots",
    project_dir: str | Path | None = None,
    create: bool = False,
) -> Path:
    """Return canonical run snapshot directory: snapshots/<run_name>/."""
    root = resolve_project_root(project_dir)
    run_dir = (
        root
        / str(base_snapshots_dir)
        / resolve_snapshot_run_name(
            run_name=run_name,
            exp_config=exp_config,
        )
    )
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_iteration_snapshot_dir(
    *,
    run_name: str | None = None,
    iteration: int,
    exp_config: Any | None = None,
    base_snapshots_dir: str = "snapshots",
    project_dir: str | Path | None = None,
    create: bool = False,
) -> Path:
    """Return canonical iteration snapshot directory: .../iteration_<n>/."""
    run_dir = resolve_run_snapshot_dir(
        run_name=run_name,
        exp_config=exp_config,
        base_snapshots_dir=base_snapshots_dir,
        project_dir=project_dir,
        create=create,
    )
    iteration_dir = run_dir / f"iteration_{int(iteration)}"
    if create:
        iteration_dir.mkdir(parents=True, exist_ok=True)
    return iteration_dir
