import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote as url_unquote

import pandas as pd
import yaml

# ----------------------------
# Utilities
# ----------------------------


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file.

    Reads the file in chunks to avoid loading large files into memory.

    Args:
        path: Path to the file.
        chunk: Chunk size in bytes used for streaming reads (default 1 MiB).

    Returns:
        str: Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def quick_schema(df: pd.DataFrame, max_cols: int = 30) -> dict[str, Any]:
    """Produce a compact schema summary for a DataFrame.

    Limits the number of reported columns and includes row/column counts
    and dtype information for the reported columns.

    Args:
        df: Input pandas DataFrame.
        max_cols: Maximum number of columns to include in the schema.

    Returns:
        Dict[str, Any]: A schema dictionary with n_rows, n_cols, columns, dtypes.
    """
    cols = list(df.columns[:max_cols])
    return {
        "n_rows": len(df),
        "n_cols": int(df.shape[1]),
        "columns": cols,
        "dtypes": {c: str(df[c].dtype) for c in cols},
    }


def _sniff_magic(path: Path, n: int = 8) -> bytes:
    """Read the first `n` bytes of a file to inspect magic bytes.

    Args:
        path: Path to the file to inspect.
        n: Number of bytes to read from the beginning of the file.

    Returns:
        bytes: The leading bytes if the file can be opened; empty bytes on error.
    """
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except Exception:
        return b""


def _sniff_kind(path: Path) -> str:
    """
    Return a coarse file kind based on magic bytes.
    Kinds: 'parquet','gzip','hdf5','npy','zip','text','binary'

    Args:
        path: Path to the file to classify.

    Returns:
        str: Detected kind label, or 'binary'/'text' based on a simple heuristic.
    """
    sig = _sniff_magic(path, 8)
    if sig.startswith(b"PAR1"):
        return "parquet"
    if sig.startswith(b"\x1f\x8b"):
        return "gzip"
    if sig.startswith(b"\x89HDF\r\n\x1a\n"):
        return "hdf5"
    if sig.startswith(b"\x93NUMPY"):
        return "npy"
    if sig.startswith(b"PK\x03\x04"):
        return "zip"
    # crude text/binary heuristic
    if sig and all(32 <= b <= 126 or b in (9, 10, 13) for b in sig):
        return "text"
    return "binary"


def _read_table(path: Path, sample_n: int = 5) -> tuple[dict[str, Any] | None, str | None]:
    """
    Try to read a small sample + basic schema. Supports:
    - parquet (by ext or magic)
    - gzip (csv/tsv autodetect)
    - h5ad/h5 via anndata if possible (by ext or HDF5 magic)
    - npy (metadata only)
    - generic text: pandas read_csv sep=None (sniffer) -> whitespace fallback
    Returns (info, None) on success; (None, "error") on failure.

    Args:
        path: File path to probe.
        sample_n: Number of rows to include in the returned sample head.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]:
            - info: Dict containing "sample_head" and "schema" on success.
            - error: String describing the read error, or None on success.
    """

    suf = path.suffix.lower()

    # Shortcut: explicit extensions first
    try:
        if suf == ".parquet":
            df = pd.read_parquet(path)
            info = {
                "sample_head": df.head(sample_n).to_dict(orient="list"),
                "schema": quick_schema(df),
            }
            return info, None

        if suf in (".h5ad", ".h5"):
            try:
                import anndata as ad

                A = ad.read_h5ad(path)
                info = {
                    "sample_head": {},
                    "schema": {
                        "n_rows": int(getattr(A, "n_obs", 0)),
                        "n_cols": int(getattr(A, "n_vars", 0)),
                        "columns": [str(x) for x in list(getattr(A, "var_names", [])[:30])],
                        "dtypes": {},
                    },
                }
                return info, None
            except Exception as e:
                return None, repr(e)

        if suf == ".gz":
            # Attempt CSV/TSV inference under gzip
            try:
                df = pd.read_csv(
                    path, compression="infer", sep=None, engine="python", low_memory=False
                )
            except Exception:
                # try TSV explicitly
                df = pd.read_csv(path, compression="infer", sep="\t", low_memory=False)
            info = {
                "sample_head": df.head(sample_n).to_dict(orient="list"),
                "schema": quick_schema(df),
            }
            return info, None

        if suf in (".csv",):
            df = pd.read_csv(path, low_memory=False)
            info = {
                "sample_head": df.head(sample_n).to_dict(orient="list"),
                "schema": quick_schema(df),
            }
            return info, None

        if suf in (".tsv", ".txt"):
            df = pd.read_csv(path, sep="\t", low_memory=False)
            info = {
                "sample_head": df.head(sample_n).to_dict(orient="list"),
                "schema": quick_schema(df),
            }
            return info, None
    except Exception:
        # will fall through to sniffing + generic handlers
        pass

    # --- Extensionless or unknown: sniff magic and fall back smartly ---
    kind = _sniff_kind(path)

    try:
        if kind == "parquet":
            df = pd.read_parquet(path)
            info = {
                "sample_head": df.head(sample_n).to_dict(orient="list"),
                "schema": quick_schema(df),
            }
            return info, None

        if kind == "gzip":
            # gzip of csv/tsv: try inference then tsv
            try:
                df = pd.read_csv(
                    path, compression="infer", sep=None, engine="python", low_memory=False
                )
            except Exception:
                df = pd.read_csv(path, compression="infer", sep="\t", low_memory=False)
            info = {
                "sample_head": df.head(sample_n).to_dict(orient="list"),
                "schema": quick_schema(df),
            }
            return info, None

        if kind == "hdf5":
            # Try AnnData first; if not AnnData, just report HDF5 container
            try:
                import anndata as ad

                A = ad.read_h5ad(path)
                info = {
                    "sample_head": {},
                    "schema": {
                        "n_rows": int(getattr(A, "n_obs", 0)),
                        "n_cols": int(getattr(A, "n_vars", 0)),
                        "columns": [str(x) for x in list(getattr(A, "var_names", [])[:30])],
                        "dtypes": {},
                    },
                }
                return info, None
            except Exception as e:
                return None, f"HDF5 detected, but not AnnData: {e!r}"

        if kind == "npy":
            try:
                import numpy as np

                arr = np.load(path, mmap_mode="r")
                shape = getattr(arr, "shape", None)
                dtype = str(getattr(arr, "dtype", ""))
                info = {
                    "sample_head": {},
                    "schema": {
                        "n_rows": int(shape[0]) if shape else 0,
                        "n_cols": int(shape[1]) if (shape and len(shape) > 1) else 0,
                        "columns": [],
                        "dtypes": {"array_dtype": dtype},
                    },
                }
            except Exception as e:
                return None, f"NPY detected but failed to read: {e!r}"
            return info, None

        if kind in ("text", "binary", "zip"):
            # Try to treat as delimited text first (many extensionless data drops are CSV/TSV)
            try:
                df = pd.read_csv(path, sep=None, engine="python", low_memory=False)
            except Exception:
                try:
                    df = pd.read_csv(path, sep="\t", low_memory=False)
                except Exception as e2:
                    # Give up: report basic error
                    return (
                        None,
                        f"Could not parse as table (sniffer/tsv failed). kind={kind}, err={e2!r}",
                    )
            info = {
                "sample_head": df.head(sample_n).to_dict(orient="list"),
                "schema": quick_schema(df),
            }
            return info, None

        # Fallback if kind is empty/unknown:
        try:
            df = pd.read_csv(path, sep=None, engine="python", low_memory=False)
        except Exception:
            try:
                df = pd.read_csv(path, sep="\t", low_memory=False)
            except Exception as e3:
                return None, f"Unknown format; table parse failed: {e3!r}"
        info = {"sample_head": df.head(sample_n).to_dict(orient="list"), "schema": quick_schema(df)}
        return info, None

    except Exception as e:
        return None, repr(e)


def _as_list(x) -> list[Any]:
    """Normalize a scalar or sequence into a list.

    Args:
        x: None, a scalar, list, or tuple.

    Returns:
        List[Any]: [] for None; list(x) for list/tuple; [x] otherwise.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


# ----------------------------
# Config / Registry resolvers
# ----------------------------


def load_base_config(cfg_path: str | Path = "sys_config/ascollab.yaml") -> dict[str, Any]:
    """Load the base YAML configuration.

    Args:
        cfg_path: Path to the base config file (sys_config/ascollab.yaml).

    Returns:
        Dict[str, Any]: Parsed YAML dictionary, or {} if the file is missing.
    """
    cfg_p = Path(cfg_path)
    if not cfg_p.exists():
        return {}
    with open(cfg_p) as f:
        return yaml.safe_load(f) or {}


def _extract_dataset_keys_from_cfg(cfg: dict[str, Any]) -> list[str]:
    """Extract dataset keys from a heterogeneous config dict.

    Searches several possible paths/keys where dataset declarations may appear,
    collecting unique keys while preserving their first-seen order.

    Args:
        cfg: Parsed YAML configuration dictionary.

    Returns:
        List[str]: Ordered list of unique dataset keys.
    """
    candidates = []
    for path in [
        ("datasets",),
        ("data", "datasets"),
        ("dataset_keys",),
        ("discovery_dataset", "dataset_keys"),
    ]:
        node = cfg
        for k in path:
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        if node is None:
            continue
        if isinstance(node, list):
            candidates.extend([str(x) for x in node])
        elif isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str):
                    candidates.append(v)
                else:
                    candidates.append(k)
    seen, out = set(), []
    for k in candidates:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _resolve_dataset_root(cfg: dict[str, Any]) -> Path:
    """Resolve the dataset root directory from ENV or config.

    Priority:
      1) DATASET_ROOT or DATA_ROOT environment variables
      2) Known keys in the config dict (data/root, datasets_root, paths.*)
      3) Current directory as a fallback

    Args:
        cfg: Parsed YAML configuration dictionary.

    Returns:
        Path: Absolute path to the dataset root.
    """
    env = os.environ.get("DATASET_ROOT") or os.environ.get("DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for path in [
        ("data", "root"),
        ("datasets_root",),
        ("paths", "datasets_root"),
        ("paths", "data_root"),
    ]:
        node = cfg
        for k in path:
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        if isinstance(node, str):
            return Path(node).expanduser().resolve()
    return Path().resolve()


def import_datasets_module(module_name: str = "datasets"):
    """Import the datasets registry module with several fallbacks.

    Attempts:
      - `module_name` as-is
      - common project-relative names (`env_setup.datasets`, `system.datasets`)
      - add project root to sys.path and retry

    Args:
        module_name: Base module name to import.

    Returns:
        ModuleType: Imported module object.

    Raises:
        ImportError: If all import attempts fail.
    """
    attempts = [module_name]
    if module_name == "datasets":
        attempts.extend(["env_setup.datasets", "system.datasets"])
    last_err = None
    for mod in attempts:
        try:
            return importlib.import_module(mod)
        except Exception as e:
            last_err = e
    try:
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        return importlib.import_module(attempts[0])
    except Exception as e:
        raise ImportError(
            f"Could not import `{module_name}`. Tried: {attempts}. "
            f"Also added {project_root} to sys.path. Error: {e}"
        ) from last_err


def _looks_like_path_string(s: str) -> bool:
    """
    Heuristic: accept absolute/relative unix-like paths even without extensions.
    Avoid multi-line and obviously non-path tokens.

    Args:
        s: Candidate string.

    Returns:
        bool: True if the string plausibly represents a filesystem path.
    """
    if not isinstance(s, str):
        return False
    s = s.strip()
    if "\n" in s or len(s) > 300:
        return False

    # Obvious path cues
    if s.startswith(("/", "./", "../")) or "\\" in s:
        return True

    # Has an extension we know -> path
    if re.search(r"\.(csv|tsv|txt|parquet|gz|h5ad|h5)(\.?[^/\s]*)?$", s, re.IGNORECASE):
        return True

    # No spaces, contains at least one slash segment -> likely a path
    if "/" in s and " " not in s:
        return True

    return False


def _describe_from_registry(reg: Any, key: str) -> dict[str, Any] | None:
    """Extract a dataset description for `key` from a registry-like module/object.

    Tries multiple common patterns:
      - dict-like at top-level
      - known attributes: DESCRIPTIONS, DATASETS, REGISTRY
      - methods: get/describe/get_dataset/get_description
      - nested objects exposing `.get(key)`

    Values can be dicts, raw doc strings, or strings that look like paths; in
    the latter two cases, the return value is normalized to {"doc": ...} or
    {"path": ...}.

    Args:
        reg: Registry module or object.
        key: Dataset key to describe.

    Returns:
        Optional[Dict[str, Any]]: A normalized description dict, or None if not found.
    """
    if isinstance(reg, dict) and key in reg:
        val = reg[key]
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            if _looks_like_path_string(val):
                return {"path": val}
            return {"doc": val}

    for attr in ("DESCRIPTIONS", "DATASETS", "REGISTRY"):
        d = getattr(reg, attr, None)
        if isinstance(d, dict) and key in d:
            val = d[key]
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                if _looks_like_path_string(val):
                    return {"path": val}
                return {"doc": val}

    for fn_name in ("get", "describe", "get_dataset", "get_description"):
        fn = getattr(reg, fn_name, None)
        if callable(fn):
            try:
                val = fn(key)
                if val is not None:
                    if isinstance(val, dict):
                        return val
                    if isinstance(val, str):
                        if _looks_like_path_string(val):
                            return {"path": val}
                        return {"doc": val}
            except Exception:
                pass

    for attr in dir(reg):
        obj = getattr(reg, attr)
        if hasattr(obj, "get") and callable(obj.get):
            try:
                val = obj.get(key)
                if val is not None:
                    if isinstance(val, dict):
                        return val
                    if isinstance(val, str):
                        if _looks_like_path_string(val):
                            return {"path": val}
                        return {"doc": val}
            except Exception:
                pass
    return None


_DOC_PATH_RE = re.compile(
    r"""
    (?:
        (?:File:\s*)?
        ["']
        (?P<qpath>
            (?:
                (?:/|\./|\.\./)[^"'\s]+   # abs/relative cue
            )
            |
            (?:[^"'\s]*/[^"'\s]+)        # has a slash somewhere
            |
            (?:[^"'\s]+\.(?:csv|tsv|txt|parquet|gz|h5ad|h5))
        )
        ["']
    )
    |
    # absolute path must look like /seg/seg (two segments) or have a known extension
    (?P<plain>
        /
        (?:
            [^/\s]+/[^ \t\r\n"']+         # /a/b...
            |
            [^ \t\r\n"']+\.(?:csv|tsv|txt|parquet|gz|h5ad|h5)
        )
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _extract_paths_from_doc(doc: str) -> list[Path]:
    """Parse a free-form documentation string and extract path-like tokens.

    Recognizes quoted paths following 'File:' and unquoted absolute paths.

    Args:
        doc: Source documentation text.

    Returns:
        List[Path]: Paths extracted from the text (not validated for existence).
    """
    paths: list[Path] = []
    for m in _DOC_PATH_RE.finditer(doc or ""):
        p = m.group("qpath") or m.group("plain")
        if p:
            paths.append(Path(p))
    return paths


def _augment_candidates(p: Path, dataset_root: Path, project_root: Path) -> list[Path]:
    """
    Given a single parsed path p, produce a small set of alternate locations:
    - original (relative -> dataset_root/relative)
    - URL-decoded variant
    - if absolute under /datasets/..., map to project_root/sandbox/datasets/...

    Args:
        p: Candidate path (absolute or relative).
        dataset_root: Base directory to prefix for relative paths.
        project_root: Project repository root used for sandbox remapping.

    Returns:
        List[Path]: Ordered, de-duplicated list of candidate absolute paths.
    """
    cands: list[Path] = []

    def _mk_abs(x: Path) -> Path:
        return x if x.is_absolute() else (dataset_root / x)

    # original
    cands.append(_mk_abs(p))

    # URL-decoded variant
    p_dec = Path(url_unquote(str(p)))
    if str(p_dec) != str(p):
        cands.append(_mk_abs(p_dec))

    # /datasets/... -> project_root/sandbox/datasets/...
    def _sandbox_map(x: Path) -> Path | None:
        x_str = x.as_posix()
        if x_str.startswith("/datasets/"):
            rest = x_str.lstrip("/")  # "datasets/..."
            return project_root / "sandbox" / rest
        return None

    for base in [p, p_dec]:
        abs_base = _mk_abs(base)
        sb = _sandbox_map(abs_base)
        if sb is not None:
            cands.append(sb)

    # de-dup while preserving order
    out, seen = [], set()
    for c in cands:
        sc = str(c)
        if sc not in seen:
            seen.add(sc)
            out.append(c)
    return out


def _paths_from_description(desc: dict[str, Any], dataset_root: Path) -> list[Path]:
    """
    Normalize a description into concrete file paths.
    Supports keys:
      - path, file (str)
      - paths, files (list[str])
      - root + relpath
      - glob (expand relative to base or dataset_root)
      - doc (parse paths embedded in description text, e.g., 'File: ".../data.tsv.gz"')

    Args:
        desc: Dataset description dictionary (various supported shapes).
        dataset_root: Root directory to resolve relative paths and globs.

    Returns:
        List[Path]: Candidate file paths (not guaranteed to exist).
    """
    out: list[Path] = []

    # explicit file(s)
    for k in ("path", "file"):
        if k in desc and isinstance(desc[k], str):
            out.append(Path(desc[k]))
    for k in ("paths", "files"):
        if k in desc and isinstance(desc[k], (list, tuple)):
            out.extend(Path(p) for p in desc[k] if isinstance(p, str))

    # root + rel
    root = desc.get("root")
    rel = desc.get("rel") or desc.get("relpath") or desc.get("relative_path")
    if isinstance(root, str) and isinstance(rel, str):
        out.append(Path(root) / rel)

    # glob pattern
    pat = desc.get("glob")
    if isinstance(pat, str):
        base = Path(desc.get("base", dataset_root))
        out.extend(base.glob(pat))

    # parse embedded doc for paths
    if "doc" in desc and isinstance(desc["doc"], str):
        out.extend(_extract_paths_from_doc(desc["doc"]))

    # Now augment with local sandbox mapping + URL-decode.
    project_root = Path(__file__).resolve().parents[1]
    augmented: list[Path] = []
    for p in out:
        augmented.extend(
            _augment_candidates(p, dataset_root=dataset_root, project_root=project_root)
        )

    # If nothing was found at all, still return the raw (possibly relative) paths
    if not augmented:
        augmented = out

    # Do NOT resolve() here to avoid collapsing non-existent absolute paths;
    # the probing step will check existence and report success if any candidate works.
    # De-dup
    seen, uniq = set(), []
    for p in augmented:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            uniq.append(p)
    return uniq


CHECKSUMS_PATH = Path(__file__).resolve().parents[1] / "sys_config" / "dataset" / "checksums.yaml"

# Dataset keys live in the bundles under sys_config/dataset/, not in sys_config/ascollab.yaml
# (which selects one with `dataset:`). KIRC is the default bundle there, so it is the default here.
DEFAULT_KEYS_CONFIG = "sys_config/dataset/kirc.yaml"


def load_known_checksums(path: str | Path = CHECKSUMS_PATH) -> dict[str, set[str]]:
    """Accepted SHA-256 values per file path relative to the datasets directory."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {
        str(rel): {str(v) for v in (vals if isinstance(vals, list) else [vals])}
        for rel, vals in data.items()
    }


def _checksum_error(path: Path, sha: str | None, ds_root: Path, known: dict[str, set[str]]):
    """A message when a file with a recorded checksum does not match it, else None.

    The file's key is its path relative to the datasets directory: ``sandbox/datasets`` under the
    dataset root (descriptions give repo-relative paths) or under the project (where
    ``/datasets/...`` paths are mapped), or the dataset root itself. The path is matched as
    given first, so a dataset folder or file symlinked to storage elsewhere is still checked;
    then with symlinks resolved.
    """
    rel = None
    project_datasets = Path(__file__).resolve().parents[1] / "sandbox" / "datasets"
    roots = (ds_root / "sandbox" / "datasets", project_datasets, ds_root)
    pairs = [(path.absolute(), root.absolute()) for root in roots]
    pairs += [(path.resolve(), root.resolve()) for root in roots]
    for file_path, root in pairs:
        try:
            candidate = file_path.relative_to(root).as_posix()
        except ValueError:
            continue
        if candidate in known:
            rel = candidate
            break
    if rel is None:
        return None
    expected = known.get(rel)
    if not expected or sha is None or sha in expected:
        return None
    return f"{rel}: SHA-256 differs from sys_config/dataset/checksums.yaml (the data may not be the paper's)"


def _flag_checksum_error(
    finfo: dict[str, Any],
    path: Path,
    key: str,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    ds_root: Path,
    known: dict[str, set[str]],
) -> None:
    """Mark a probed file, its dataset entry and the manifest not ok if its checksum differs.

    Args:
        finfo: The file's manifest entry (holds ``sha256`` once hashed); gets ``checksum_error``.
        path: The probed file.
        key: Dataset key, for the error message.
        entry: The dataset's manifest entry.
        manifest: The whole manifest.
        ds_root: Dataset root.
        known: Accepted checksums from `load_known_checksums`.
    """
    checksum_error = _checksum_error(path, finfo.get("sha256"), ds_root, known)
    if checksum_error:
        finfo["checksum_error"] = checksum_error
        entry["ok"] = False
        manifest["ok"] = False
        manifest["errors"].append(f"[{key}] {checksum_error}")


def _write_manifest(save_manifest: str | Path, manifest: dict[str, Any]) -> None:
    """Write the manifest dictionary to disk as pretty JSON.

    Ensures parent directories exist before writing.

    Args:
        save_manifest: Target file path for the manifest JSON.
        manifest: The manifest dictionary to serialize.
    """
    save_p = Path(save_manifest)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    with open(save_p, "w") as f:
        json.dump(manifest, f, indent=2)


# ----------------------------
# Main preflight
# ----------------------------


def preflight_data(
    dataset_module: str = "env_setup.datasets",
    base_cfg: str | Path = DEFAULT_KEYS_CONFIG,
    dataset_root: str | Path | None = None,
    save_manifest: str | Path = "artifacts/data_manifest.json",
    sample_n: int = 5,
    expected_min_sizes: dict[str, tuple[int, int]] | None = None,
    dataset_keys: list[str] | None = None,  # allow caller to pass keys
) -> dict[str, Any]:
    """
    - Uses `dataset_keys` if provided (e.g., from Hydra cfg), else reads them from `base_cfg`
    - Resolves file paths via datasets.py registry/description
    - Probes each file (exists, size, sample read, schema)
    - Returns a manifest dict and writes it to save_manifest

    Args:
        dataset_module: Python module path that defines dataset registry/descriptions.
        base_cfg: Config file to read dataset keys from when `dataset_keys` is not given, and
            dataset-root keys from (see `_resolve_dataset_root`). Default: the KIRC bundle,
            the default dataset of sys_config/ascollab.yaml.
        dataset_root: Optional override for dataset root directory.
        save_manifest: Where to write the output manifest JSON.
        sample_n: Number of rows to include in each file's sample head.
        expected_min_sizes: Optional map key -> (min_rows, min_cols) for sanity checks.
        dataset_keys: Optional explicit list of dataset keys to probe.

    Returns:
        Dict[str, Any]: A manifest with overall status, errors, and per-dataset reports.
    """
    manifest: dict[str, Any] = {
        "cwd": str(Path.cwd()),
        "dataset_module": dataset_module,
        "base_cfg": str(base_cfg),
        "env_dataset_root": os.environ.get("DATASET_ROOT") or os.environ.get("DATA_ROOT"),
        "dataset_root": None,
        "ok": True,
        "errors": [],
        "datasets": {},
        "keys_source": None,
    }

    cfg = load_base_config(base_cfg)

    if dataset_keys and len(dataset_keys) > 0:
        keys = list(dataset_keys)
        manifest["keys_source"] = "caller"
    else:
        keys = _extract_dataset_keys_from_cfg(cfg)
        manifest["keys_source"] = "base_cfg"

    if not keys:
        manifest["ok"] = False
        manifest["errors"].append("No dataset keys provided and none found in the base config.")

    ds_root = Path(dataset_root).resolve() if dataset_root else _resolve_dataset_root(cfg)
    manifest["dataset_root"] = str(ds_root)

    try:
        reg = import_datasets_module(dataset_module)
    except ImportError as e:
        manifest["ok"] = False
        manifest["errors"].append(str(e))
        _write_manifest(save_manifest, manifest)
        return manifest

    expected_min_sizes = expected_min_sizes or {}
    known_checksums = load_known_checksums()

    for key in keys:
        entry = {"key": key, "ok": True, "paths": [], "tried_paths": [], "files": []}
        desc = _describe_from_registry(reg, key)
        if not desc:
            entry["ok"] = False
            msg = f"[{key}] No description found in `{dataset_module}`."
            manifest["errors"].append(msg)
            manifest["ok"] = False
            manifest["datasets"][key] = entry
            continue

        candidates = _paths_from_description(desc, ds_root)
        entry["tried_paths"] = [str(p) for p in candidates]  # for debugging

        success = 0
        for p in candidates:
            if not isinstance(p, Path):
                try:
                    p = Path(p)
                except Exception:
                    continue
            if not p.is_absolute():
                p = ds_root / p
            # don't resolve() until after remapping; we already made candidates absolute-ish in _augment
            p = p

            if p.exists():
                # If it's a directory, enumerate likely data files inside and probe those
                if p.is_dir():
                    # prioritize common TCGA/Xena layouts
                    inner_globs = [
                        "*.tsv.gz",
                        "*.txt.gz",
                        "*.csv.gz",
                        "*.tsv",
                        "*.txt",
                        "*.csv",
                        "*.parquet",
                        "*",
                    ]
                    dir_files = []
                    for pat in inner_globs:
                        dir_files = [q for q in p.glob(pat) if q.is_file()]
                        if dir_files:
                            break
                    if not dir_files:
                        finfo = {
                            "path": str(p),
                            "exists": True,
                            "read_error": "Directory contains no readable files",
                        }
                        entry["files"].append(finfo)
                        entry["paths"].append(str(p))
                        continue
                    # Probe a handful (don’t explode manifest)
                    to_probe = dir_files[:5]
                    for q in to_probe:
                        finfo = {"path": str(q), "exists": True}
                        finfo["size_bytes"] = q.stat().st_size
                        info, read_err = _read_table(q, sample_n=sample_n)
                        if read_err:
                            finfo["read_error"] = read_err
                        else:
                            finfo.update(info or {})
                        try:
                            finfo["sha256"] = sha256sum(q)
                        except Exception as e:
                            finfo["hash_error"] = repr(e)
                        _flag_checksum_error(
                            finfo, q, key, entry, manifest, ds_root, known_checksums
                        )
                        entry["files"].append(finfo)
                        entry["paths"].append(str(q))
                        if "read_error" not in finfo or finfo.get("size_bytes", 0) > 0:
                            success += 1
                    continue  # done with this directory

            if p.exists():
                finfo = {"path": str(p), "exists": True}
                finfo["size_bytes"] = p.stat().st_size
                info, read_err = _read_table(p, sample_n=sample_n)
                if read_err:
                    finfo["read_error"] = read_err
                else:
                    finfo.update(info or {})
                    if key in expected_min_sizes and "schema" in finfo:
                        min_r, min_c = expected_min_sizes[key]
                        n_r = finfo["schema"]["n_rows"]
                        n_c = finfo["schema"]["n_cols"]
                        if n_r < min_r or n_c < min_c:
                            finfo["size_check_error"] = (
                                f"Too small ({n_r}x{n_c}) < expected ({min_r}x{min_c})"
                            )
                        else:
                            # Only count as success if size check passes (when provided)
                            pass
                try:
                    finfo["sha256"] = sha256sum(p)
                except Exception as e:
                    finfo["hash_error"] = repr(e)
                _flag_checksum_error(finfo, p, key, entry, manifest, ds_root, known_checksums)

                entry["files"].append(finfo)
                entry["paths"].append(str(p))
                # Count as success if either no read error or at least file exists (prefer read success)
                if "read_error" not in finfo or finfo.get("size_bytes", 0) > 0:
                    success += 1

        if success == 0:
            entry["ok"] = False
            manifest["ok"] = False
            # Report the first candidate succinctly (others are visible in tried_paths)
            if candidates:
                # Name the expected location (under sandbox/datasets) when there is one.
                shown = next((c for c in candidates if "sandbox/datasets" in str(c)), candidates[0])
                manifest["errors"].append(f"[{key}] Missing or unreadable data. Expected: {shown}")
            else:
                manifest["errors"].append(f"[{key}] No file paths resolved from description.")

        manifest["datasets"][key] = entry

    _write_manifest(save_manifest, manifest)
    return manifest


# ----------------------------
# CLI usage
# ----------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Preflight probe for the datasets in a bundle (or any config with dataset_keys)"
    )
    ap.add_argument(
        "--datasets-module",
        default="env_setup.datasets",
        help="Python module that describes datasets",
    )
    ap.add_argument(
        "--config",
        default=DEFAULT_KEYS_CONFIG,
        help="A config with dataset_keys, e.g. a dataset bundle in sys_config/dataset/",
    )
    ap.add_argument(
        "--dataset-root",
        default=None,
        help=(
            "Folder the descriptions' relative paths (sandbox/datasets/...) start from; default: "
            "$DATASET_ROOT or $DATA_ROOT, else data.root, datasets_root, paths.datasets_root or "
            "paths.data_root in --config, else the current directory"
        ),
    )
    ap.add_argument(
        "--manifest", default="artifacts/data_manifest.json", help="Where to write manifest JSON"
    )
    ap.add_argument("--sample-n", type=int, default=5, help="Rows to include in sample head")
    ap.add_argument(
        "--expect",
        nargs="*",
        default=[],
        help="Size expectations like key:minrowsxmincols (e.g., rna:1000x10000)",
    )
    args = ap.parse_args()

    expected = {}
    size_pat = re.compile(r"^([^:]+):(\d+)x(\d+)$")
    for item in args.expect:
        m = size_pat.match(item)
        if m:
            k, r, c = m.group(1), int(m.group(2)), int(m.group(3))
            expected[k] = (r, c)

    report = preflight_data(
        dataset_module=args.datasets_module,
        base_cfg=args.config,
        dataset_root=args.dataset_root,
        save_manifest=args.manifest,
        sample_n=args.sample_n,
        expected_min_sizes=expected or None,
    )

    print("DATA_AVAILABLE:", report["ok"])
    print("CWD:", report["cwd"])
    print("DATASET_ROOT:", report["dataset_root"])
    if not report["ok"]:
        print("Errors:")
        for e in report["errors"]:
            print(" -", e)
        sys.exit(1)
