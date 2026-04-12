"""App-wide settings loaded from environment or defaults."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    title: str = "whaleshark-reid (dev)"
    host: str = "0.0.0.0"
    port: int = 8090
    db_path: Path = Path("cache/state.db")
    cache_dir: Path = Path("cache/")
    user: str = "dev"
    max_queue_page_size: int = 100

    model_config = {"env_prefix": "WHALESHARK_"}
