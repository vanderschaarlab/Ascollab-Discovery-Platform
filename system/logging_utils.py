import io
import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_file_logging_hydra_safe(
    filename: str = "run.log",
    level: int = logging.INFO,
    max_bytes: int = 10_000_000,
    backups: int = 3,
) -> str:
    """Attach a rotating file handler to Hydra's root logger without replacing existing handlers.

    Args:
        filename: Log file name.
        level: Logging level for the file handler.
        max_bytes: Maximum size before rotation.
        backups: Number of backup files to keep.

    Returns:
        Absolute path to the log file.
    """
    logger = logging.getLogger()  # root
    # Determine Hydra run dir if available
    try:
        from hydra.core.hydra_config import HydraConfig

        run_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        run_dir = os.getcwd()
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, filename)

    # Avoid duplicate handler if called twice
    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler) and getattr(h, "_our_handler", False):
            return log_path

    fh = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    fh.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    fh._our_handler = True  # mark to avoid dup
    logger.addHandler(fh)

    # keep console clean at INFO; Hydra usually already has console handlers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.captureWarnings(True)  # route warnings to logging
    return log_path


class _StreamToLogger(io.TextIOBase):
    """File-like object that writes incoming text to a logger and optionally mirrors to a real stream."""

    def __init__(self, logger: logging.Logger, level: int, mirror: io.TextIOBase | None = None):
        self.logger = logger
        self.level = level
        self.mirror = mirror
        self._buffer = ""

    def write(self, buf: str) -> int:
        if self.mirror is not None:
            try:
                self.mirror.write(buf)
            except Exception:
                pass
        # Buffer until newline to avoid splitting lines mid-message
        self._buffer += buf
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self.logger.log(self.level, line)
        return len(buf)

    def flush(self) -> None:
        if self.mirror is not None:
            try:
                self.mirror.flush()
            except Exception:
                pass
        if self._buffer:
            self.logger.log(self.level, self._buffer.rstrip("\r"))
            self._buffer = ""

    # Make it behave like a normal stream for most libs
    def isatty(self) -> bool:
        return False


def tee_prints_to_logging(
    mirror_stdout: bool = True, mirror_stderr: bool = True
) -> tuple[io.TextIOBase, io.TextIOBase]:
    """Redirect stdout/stderr to logging, optionally mirroring to original streams.

    Args:
        mirror_stdout: Whether to also write stdout to the original stream.
        mirror_stderr: Whether to also write stderr to the original stream.

    Returns:
        Tuple of (orig_stdout, orig_stderr) for restoration.
    """
    root = logging.getLogger()
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _StreamToLogger(
        logging.getLogger("STDOUT"), level=logging.INFO, mirror=orig_out if mirror_stdout else None
    )
    sys.stderr = _StreamToLogger(
        logging.getLogger("STDERR"), level=logging.ERROR, mirror=orig_err if mirror_stderr else None
    )
    # Ensure root has at least one handler (Hydra usually does)
    if not root.handlers:
        logging.basicConfig(level=logging.INFO)
    return orig_out, orig_err


def restore_prints(orig_out: io.TextIOBase, orig_err: io.TextIOBase) -> None:
    """Restore stdout/stderr after teeing to logging."""
    sys.stdout = orig_out
    sys.stderr = orig_err
