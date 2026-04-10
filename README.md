# whaleshark-reid

Whale shark re-identification pipeline and dev-mode review app.

Design specs live in `docs/superpowers/specs/`. Implementation plans live in `docs/superpowers/plans/`.

## Development

Install in editable mode:

    pip install -e ".[dev]"

Run tests:

    pytest -x

## Layout

- `src/whaleshark_reid/core/` — pure-Python core (schema, ingest, embed, cluster, matching, feedback, metrics)
- `src/whaleshark_reid/storage/` — SQLite + parquet state
- `tests/` — unit + integration tests
- `docs/superpowers/specs/` — design specs
- `docs/superpowers/plans/` — implementation plans
