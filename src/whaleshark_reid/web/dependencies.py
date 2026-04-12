"""FastAPI dependency injection — Storage and Settings singletons."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.settings import Settings


@lru_cache(maxsize=1)
def _get_settings() -> Settings:
    return Settings()


_storage_instance: Storage | None = None


def get_settings() -> Settings:
    return _get_settings()


def get_storage() -> Storage:
    global _storage_instance
    if _storage_instance is None:
        settings = get_settings()
        _storage_instance = Storage(settings.db_path)
        _storage_instance.init_schema()
    return _storage_instance


def override_storage(storage: Storage) -> None:
    """For tests: inject a pre-configured Storage instance."""
    global _storage_instance
    _storage_instance = storage


def override_settings(settings: Settings) -> None:
    """For tests: inject custom settings."""
    _get_settings.cache_clear()
    # Monkey-patch the cached getter
    import whaleshark_reid.web.dependencies as mod
    mod._get_settings = lru_cache(maxsize=1)(lambda: settings)
