# Whale Shark Re-ID — Project Overview

**Date:** 2026-04-10
**Status:** Phase 1 design approved, implementation not yet started
**Location on disk:** `/workspace/catalog-match/whaleshark-reid/`

## Goal

Build a dev-mode web application for whale shark re-identification that lets an ML engineer iterate on the matching/clustering pipeline while using the app as the labeling and review surface. The long-term product is the underlying ML engine plus a clean review UI that other researchers can use to curate their own catalogs; Phase 1 is the dog-fooding slice the ML engineer uses to improve the engine.

The core loop: ingest image catalogs → extract MiewID embeddings → cluster + propose pairwise matches → human reviews pairs in a carousel UI → derived individuals feed back into the ML pipeline → experiments are tracked and compared.

## Non-goals

- Auth or multi-user support (single-user dev tool in a container).
- Production-grade deployment (this runs on localhost via uvicorn inside a docker container; port-mapped to the host).
- Matching algorithms beyond the MiewID + DBSCAN/HDBSCAN + distance-threshold baseline. Advanced ML experiments (PairX, LightGlue, hybrid MiewID, curation techniques, scaling to 100k catalogs) live in a separate Phase 3+ track.

## Phases

### Phase 1 — iNat cold-start (this design cycle)

Scope: iNat whale shark data only, which is already downloaded to `/workspace/catalog-match/inat-download-recent-species-sightings/whaleshark_inat_v1/`. No ground-truth labels — this is the cold-start problem where the user builds up a labeled catalog by reviewing proposed matches. No metrics-vs-GT in Phase 1 because there is no GT on iNat data; the benchmark is agreement with the existing `extract_and_evaluate_whalesharks.ipynb` notebook outputs.

**Deliverables:**
- Python core package (`whaleshark_reid.core`) with data schema, ingest, embedding, clustering, matching, feedback modules
- CLI (`whaleshark_reid.cli`) that drives the pipeline stages
- Web UI (`whaleshark_reid.web`) with five pages: pair carousel, list/log view, cluster/embedding tree view, experiments view, and a stub for the Phase 2 map page
- SQLite + parquet storage
- Four-layer test suite (unit, storage, integration, web)

### Phase 2 — Wildbook ingestion + GT metrics

Scope: ingest Wildbook COCO catalogs (which carry UUIDs natively for annotations, images, and individuals), run matching in merge-into-existing mode, compute R1/mAP/calibration on the Wildbook validation subset reusing `wbia_miew_id.metrics.eval_onevsall` and `precision_at_k` and the `compute_calibration` helper. Add the dedicated full map page. Add reconcile-two-labeled-datasets mode.

Future specs in this phase:
- `04-wildbook-ingest.md`
- `05-gt-metrics-harness.md`
- `06-full-map-page.md`
- `07-reconcile-mode.md`

### Phase 3+ — Advanced ML experiments

PairX re-ranking, LightGlue-style matchers, hybrid MiewID checkpoints, curation techniques (e.g. https://arxiv.org/abs/2511.06658), scaling tricks for 20k-100k catalogs. Each experiment gets its own design + plan cycle.

Future spec: `08-advanced-matching-track.md`

## Architecture at a glance

```
                     ┌─────────────────────────────────────────────┐
                     │            whaleshark_reid.core             │
                     │  (pure Python, reusable, notebook-friendly) │
                     │                                             │
                     │  schema ┐                                   │
                     │  ingest ┤                                   │
                     │  embed  ┼─► storage (sqlite + parquet cache)│
                     │  cluster┤                                   │
                     │  match  ┤                                   │
                     │  feedback                                   │
                     └──▲──────────────────────────────────▲───────┘
                        │                                  │
            ┌───────────┴──────────┐            ┌──────────┴─────────┐
            │  whaleshark_reid.cli │            │ whaleshark_reid.web│
            │  (Typer)             │            │ (FastAPI + HTMX)   │
            │                      │            │                    │
            │  ingest, embed,      │            │  services/  ◄──── stable JSON API
            │  cluster, matching,  │            │  routes/    (HTML & JSON)
            │  project,            │            │  templates/ (Jinja + HTMX)
            │  rebuild-individuals,│            │  static/    (Plotly, Leaflet, ~50 LOC JS)
            │  run-all             │            │                    │
            └──────────────────────┘            └────────────────────┘
```

### Design invariants

1. **`core/` has zero web/CLI dependencies.** Every function takes inputs and returns outputs. Notebook-first: the same functions that the CLI and web call can be invoked from a Jupyter cell.
2. **`web/services/` is a stack-agnostic JSON-shaped service layer.** Route handlers are thin adapters. Future React swap replaces `web/templates/` + `web/static/` and adds `/api/*` JSON routes calling the same services.
3. **Storage is SQLite + parquet, not Postgres.** Tabular state in one SQLite file; embeddings, cluster labels, and projections in parquet files keyed by `run_id`.
4. **MiewID runs in-process**, not via `ml-service` HTTP. Faster, batchable, no 2-request semaphore bottleneck. Same preprocessing pipeline (`wbia_miew_id.datasets.helpers.get_chip_from_img`, ImageNet norm, 440×440 resize).
5. **All identifiers are UUIDs.** `annotation_uuid`, `image_uuid`, `name_uuid` are the keys across the system. Integer IDs from source systems (iNat `observation_id`, COCO `id`) are kept as separate traceability columns but never used as primary keys. iNat cold-start UUIDs are deterministic via `uuid5(namespace, ...)` for idempotent re-ingest.
6. **Pipeline is CLI-primary.** Each stage is a Typer command, idempotent, writes to SQLite + parquet. The web UI is a view over that state plus a "run pipeline" button that spawns CLI commands as subprocesses.
7. **Feedback model is append-only pair log + derived individuals.** Source of truth is `pair_decisions`. `annotations.name_uuid` is a materialized view computed via union-find over confirmed pairs and refreshed by `rebuild-individuals` (fresh `uuid4` per component per rebuild — users don't compare UUIDs across rebuilds).

## Nomenclature (glossary, preserved from existing code)

These names are borrowed from `wbia_miew_id`, `whaleshark_inat_v1/common.py`, and the `extract_and_evaluate_whalesharks.ipynb` notebook. New code must match them.

| Concept | Name in code | Notes |
|---------|--------------|-------|
| Embedding | `embedding`, `embeddings`, `db_emb`, `test_emb`, `q_pids` | 512-D float32 from MiewID-msv3 |
| Individual identity (machine) | `name_uuid` | UUID, primary key for individual |
| Individual identity (human label) | `name` | Display string, e.g. "J-42", "Mary" |
| Annotation | `annotation`, `annotation_uuid` | One image crop (bbox+theta) of one individual |
| Image | `image_uuid` | Source image file; multiple annotations may share one |
| Bounding box | `bbox` | `[x, y, w, h]`, NOT `[x1, y1, x2, y2]` |
| Rotation | `theta` | Radians, float |
| Viewpoint | `viewpoint` | String, e.g. "left_side", "unknown" |
| Species | `species` | Currently always "whaleshark" |
| Distance | `distance`, `distmat` | Cosine distance ∈ [0, 2], lower = better match |
| Query | `query`, `test_emb`, `q_pids` | Annotations being matched |
| Gallery | `gallery`, `db_emb`, `db_lbls` | Reference set |
| Cluster label | `cluster_label`, `dbscan_label` | Int; `-1` = noise |
| GPS | `gps_lat_captured`, `gps_lon_captured` | WGS84 decimal degrees |
| Date | `date_captured` | ISO date string |
| Source reference | `observation_id`, `photo_index`, `source_annotation_id`, `source_image_id`, `source_individual_id` | Preserved for traceability; never primary keys |
| Pipeline run | `run_id` | String `"run_<ts>_<short-hash>"` |

## Reused functions (do not rewrite)

From `wbia-plugin-miew-id`:
- `wbia_miew_id.metrics.distance.compute_distance_matrix(input1, input2, metric='cosine')`
- `wbia_miew_id.metrics.distance.compute_batched_distance_matrix(...)` (memory-efficient variant)
- `wbia_miew_id.metrics.knn.predict_k_neigh(db_emb, db_lbls, test_emb, k=5)` — kept for Phase 2 gallery matching
- `wbia_miew_id.metrics.eval_metrics.precision_at_k(names, distmat, names_db=None, ranks=...)` — Phase 2 GT metrics
- `wbia_miew_id.metrics.eval_metrics.topk_average_precision(...)` — Phase 2 GT metrics
- `wbia_miew_id.metrics.eval_metrics.compute_calibration(true_labels, pred_labels, confidences, num_bins=10)` — Phase 2 calibration
- `wbia_miew_id.metrics.eval_onevsall.eval_onevsall(distmat, q_pids, max_rank=50)` — Phase 2 R1/mAP
- `wbia_miew_id.datasets.helpers.get_chip_from_img(img, bbox, theta)` — **Phase 1 uses this directly** to crop embedding inputs
- `wbia_miew_id.datasets.default_dataset.MiewIdDataset` — reference for field names and preprocessing

From `ml-service`:
- `ml-service/app/models/miewid.py::MiewidModel.extract_embeddings(image_bytes, bbox, theta)` — reference implementation; Phase 1 replicates its preprocessing in-process rather than calling over HTTP

From existing iNat work:
- `/workspace/catalog-match/inat-download-recent-species-sightings/whaleshark_inat_v1/common.py` — utility patterns for bbox handling, plotting, stats
- `/workspace/catalog-match/whaleshark/extract_and_evaluate_whalesharks.ipynb` — benchmark notebook; DBSCAN(eps=0.7, min_samples=2, metric='cosine') on StandardScaler-normalized embeddings is the canonical clustering config. Phase 1 core must reproduce its outputs on the same data.

## Data already on disk

- `/workspace/catalog-match/inat-download-recent-species-sightings/whaleshark_inat_v1/photos/` — 1019 JPEGs
- `/workspace/catalog-match/inat-download-recent-species-sightings/whaleshark_inat_v1/inat_observations_export_Rhincodon-typus_20251120.csv` — 343 iNat observations, raw export
- `/workspace/catalog-match/whaleshark/whaleshark_inat_v1.csv` — 377 rows, minimal MiewID-ready schema (10 cols: `theta, viewpoint, name, file_name, species, file_path, x, y, w, h`). **The iNat CSV already contains pre-computed bounding boxes** (not full-image) — Phase 1 uses them directly.
- `/workspace/catalog-match/whaleshark/dfx_whaleshark_inat_v1.csv` — 377 rows, rich COCO-like schema (41 cols) with provenance fields (`observation_id`, `image_uuid`, `photographer`, `license`, `date_captured`, `gps_lat_captured`, `gps_lon_captured`, `coco_url`, etc.)
- `/workspace/catalog-match/whaleshark/df_exploded_inat_v1.csv` — 424 rows, raw iNat export exploded per photo (53 cols, has "excessive fields" to filter down)
- `/workspace/datasets/whaleshark.coco/` — Wildbook whaleshark COCO dataset (6.1 GB, train/val/test). **Phase 2 input only.** Not touched in Phase 1.

## Sub-specs

Phase 1 is decomposed into three sub-specs. They are written in execution order and cross-reference each other. **At the start of implementing each sub-spec, re-read this overview and the other sub-specs** to check that nothing has drifted since design time.

1. **[01-core-engine.md](2026-04-10-whaleshark-reid-01-core-engine.md)** — Python core: schema, storage, ingest, embed, cluster, matching, feedback, metrics. Implementation-first, depends on nothing. Validated by reproducing `extract_and_evaluate_whalesharks.ipynb` outputs from a notebook. This is the "Python core solid" checkpoint before monorepo scaffolding.

2. **[02-cli-and-orchestration.md](2026-04-10-whaleshark-reid-02-cli-and-orchestration.md)** — Typer CLI, pipeline orchestration, `RunContext`, experiment tracking wiring (`runs` + `experiments` tables). Depends on core-engine.

3. **[03-web-ui.md](2026-04-10-whaleshark-reid-03-web-ui.md)** — FastAPI app, service layer, HTMX templates, the five UI pages, keyboard shortcuts, image serving, SSE for subprocess output, test suite (Layers 1-4). Depends on core-engine. Can be built in parallel with the CLI once core is done.

## Re-evaluation checkpoint (for implementation time)

Before starting work on each sub-spec's implementation plan, verify:

1. **Re-read this overview.** Has the phase boundary shifted? Are any of the design invariants stale?
2. **Re-read the other two sub-specs.** Do any function signatures or schema fields conflict?
3. **Check disk state.** Has the iNat data or the benchmark notebook changed since 2026-04-10?
4. **Check `wbia-plugin-miew-id`.** Have any of the reused function signatures changed? If so, update the sub-spec before implementing.
5. **Check `ml-service/app/models/miewid.py`.** Has the preprocessing pipeline changed?
6. **Report deltas.** If anything has changed, flag it to the user before writing the implementation plan.

This checkpoint is mandatory. Skipping it is how spec drift happens.

## Success criteria (Phase 1)

Phase 1 is "done" when all of the following are true:

1. **Core reproduces the benchmark notebook.** Running the pipeline end-to-end on the iNat data produces the same DBSCAN cluster count and distance distribution as `extract_and_evaluate_whalesharks.ipynb` (within sklearn determinism tolerance).
2. **CLI runs end-to-end.** `whaleshark-reid run-all --source inat --csv ... --photos-dir ...` completes without errors, writes a new `runs` row per stage, and populates `annotations`, `pair_queue`, `embeddings/`, `clusters/`, `projections/`.
3. **Web UI renders all five pages.** Pair carousel, list/log view (three tabs), cluster view, experiments view, map-stub page. No Jinja errors, no 500s on fixture data.
4. **Pair review round-trip works.** Opening the carousel, pressing Y/N/Space a few times, then re-running `rebuild-individuals`, populates `annotations.name_uuid` with derived `name_uuid`s for all annotations in confirmed components. Navigating to the Individuals tab shows them grouped by `name_uuid`.
5. **Keyboard shortcuts work.** Y/N/U/Space/J/K register decisions and navigate.
6. **Map inset renders.** Leaflet panel on the carousel shows both pair annotations' GPS when both have coordinates.
7. **Experiments view compares two runs.** Running the pipeline twice with different `--eps` values produces two `runs` rows, and the diff-two-runs tool shows both the config deltas and the metric deltas.
8. **Test suite passes.** All four test layers pass. `pytest` completes under 10 seconds.
9. **No hand-written matching code for things that exist in `wbia-plugin-miew-id`.** All distance/rank/calibration calls go through the reused functions.
