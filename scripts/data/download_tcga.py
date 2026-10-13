"""Download the TCGA KIRC, LUAD and PAAD files from the UCSC Xena hub.

    python -m scripts.data.download_tcga [--cohorts kirc luad paad] [--dest DIR]

Writes each cohort's four files (RNA-seq ``HiSeqV2``, ``RPPA``, PARADIGM ``merged_reals`` and
curated survival) under ``DIR`` (default: ``$SANDBOX_DATASETS_DIR``, else ``sandbox/datasets``),
with the folder and file names the dataset descriptions in ``env_setup/datasets.py`` expect, and
checks each against the SHA-256 in ``sys_config/dataset/checksums.yaml``. Existing files with the
right checksum are kept; any other existing file (a different checksum, or none recorded) is
downloaded again and replaced. Files are saved as downloaded, not unpacked or converted. The data
are TCGA data redistributed by UCSC Xena (https://xenabrowser.net); see docs/data.md for their
terms of use.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

from env_setup.preflight_data import load_known_checksums

HUB = "https://tcga.xenahubs.net/download"

# cohort -> (folder under the datasets dir, Xena cohort code)
COHORTS = {
    "kirc": ("KIRC_NeurIPS", "KIRC"),
    "luad": ("lung_cancer", "LUAD"),
    "paad": ("PAAD", "PAAD"),
}


def files_for(cohort: str) -> list[tuple[str, str]]:
    """(Xena dataset path, local relative path) for one cohort's four files."""
    folder, code = COHORTS[cohort]
    return [
        (f"TCGA.{code}.sampleMap/HiSeqV2.gz", f"{folder}/TCGA.{code}.sampleMap_HiSeqV2.gz"),
        (f"TCGA.{code}.sampleMap/RPPA.gz", f"{folder}/TCGA.{code}.sampleMap_RPPA.gz"),
        (
            f"merge_merged_reals/{code}_merge_merged_reals.txt.gz",
            f"{folder}/merge_merged_reals_{code}_merge_merged_reals.txt.gz",
        ),
        (f"survival/{code}_survival.txt", f"{folder}/survival_F{code}_survival.txt"),
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cohorts", nargs="+", default=sorted(COHORTS), choices=sorted(COHORTS))
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("SANDBOX_DATASETS_DIR") or "sandbox/datasets"),
        help="Datasets directory (mounted as /datasets in the sandbox)",
    )
    args = parser.parse_args(argv)

    expected = load_known_checksums()
    failures = []
    for cohort in args.cohorts:
        for xena_path, rel in files_for(cohort):
            target = args.dest / rel
            want = expected.get(rel)
            if target.exists() and want and sha256(target) in want:
                print(f"[ok] {rel} (already present)")
                continue
            print(f"[get] {HUB}/{xena_path} -> {target}")
            download(f"{HUB}/{xena_path}", target)
            got = sha256(target)
            if want and got not in want:
                failures.append(rel)
                print(f"[!!] {rel}: SHA-256 {got} is not a recorded checksum", file=sys.stderr)
            elif not want:
                print(f"[??] {rel}: no checksum recorded; SHA-256 {got}")
    if failures:
        print(
            f"{len(failures)} file(s) differ from the recorded checksums; Xena may have updated "
            "them. Runs stop at the data check unless data.preflight_strict=false.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
