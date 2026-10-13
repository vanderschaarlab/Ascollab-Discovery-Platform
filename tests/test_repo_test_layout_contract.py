from pathlib import Path

from tests.isolation import tracked_files

REPO_ROOT = Path(__file__).resolve().parents[1]
# Test folders mirror these source folders; e2e and golden test the pipeline as a whole.
SOURCE_DIRS = {
    "agents",
    "env_setup",
    "notebooks",
    "ranker",
    "sandbox",
    "scripts",
    "stores",
    "system",
}
WHOLE_PIPELINE_DIRS = {"e2e", "golden"}


def _tracked() -> list[str]:
    return tracked_files(REPO_ROOT)


def test_no_pytest_modules_live_in_production_packages():
    offenders = [
        path
        for path in _tracked()
        if Path(path).name.startswith("test_")
        and path.endswith(".py")
        and not path.startswith("tests/")
    ]

    assert offenders == []


def test_test_folders_mirror_source_folders():
    test_dirs = {
        Path(path).parts[1]
        for path in _tracked()
        if path.startswith("tests/") and path.count("/") > 1
    }

    assert test_dirs <= SOURCE_DIRS | WHOLE_PIPELINE_DIRS
    assert all((REPO_ROOT / d).is_dir() for d in SOURCE_DIRS)
