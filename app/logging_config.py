"""Configure application-wide logging (console + `logs/app.log`)."""

from __future__ import annotations

import logging
from pathlib import Path

from app.utils import project_root


def configure_logging() -> None:
    """Install root handlers once (idempotent via `force=True` on basicConfig)."""
    log_dir: Path = project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
