"""FastAPI application factory."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def create_app() -> FastAPI:
    from whaleshark_reid.web.settings import Settings

    settings = Settings()

    app = FastAPI(title=settings.title, docs_url=None, redoc_url=None)
    app.state.settings = settings

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Routes will be added in subsequent tasks via app.include_router(...)
    # For now, a minimal health check:
    @app.get("/health")
    def health():
        return {"status": "ok"}

    from whaleshark_reid.web.routes import image
    app.include_router(image.router)

    from whaleshark_reid.web.routes import pairs
    app.include_router(pairs.router)

    from whaleshark_reid.web.routes import list as list_router
    app.include_router(list_router.router)

    from whaleshark_reid.web.routes import experiments as exp_router
    app.include_router(exp_router.router)

    from whaleshark_reid.web.routes import clusters
    app.include_router(clusters.router)

    return app
