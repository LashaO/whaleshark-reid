PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS annotations (
    annotation_uuid TEXT PRIMARY KEY,
    image_uuid TEXT NOT NULL,
    name_uuid TEXT,

    source TEXT NOT NULL,
    source_annotation_id TEXT,
    source_image_id TEXT,
    source_individual_id TEXT,
    observation_id INTEGER,
    photo_index INTEGER,

    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    bbox_x REAL NOT NULL,
    bbox_y REAL NOT NULL,
    bbox_w REAL NOT NULL,
    bbox_h REAL NOT NULL,
    theta REAL NOT NULL DEFAULT 0.0,
    viewpoint TEXT NOT NULL DEFAULT 'unknown',
    species TEXT NOT NULL DEFAULT 'whaleshark',
    name TEXT,

    photographer TEXT,
    license TEXT,
    date_captured TEXT,
    gps_lat_captured REAL,
    gps_lon_captured REAL,
    coco_url TEXT,
    flickr_url TEXT,
    height INTEGER,
    width INTEGER,
    conf REAL,
    quality_grade TEXT,

    name_viewpoint TEXT,
    species_viewpoint TEXT,

    ingested_run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(source, observation_id, photo_index)
);
CREATE INDEX IF NOT EXISTS idx_ann_source ON annotations(source);
CREATE INDEX IF NOT EXISTS idx_ann_obs ON annotations(observation_id);
CREATE INDEX IF NOT EXISTS idx_ann_image_uuid ON annotations(image_uuid);
CREATE INDEX IF NOT EXISTS idx_ann_name_uuid ON annotations(name_uuid) WHERE name_uuid IS NOT NULL;

CREATE TABLE IF NOT EXISTS pair_decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ann_a_uuid TEXT NOT NULL REFERENCES annotations(annotation_uuid),
    ann_b_uuid TEXT NOT NULL REFERENCES annotations(annotation_uuid),
    decision TEXT NOT NULL CHECK(decision IN ('match','no_match','skip','unsure')),
    distance REAL,
    run_id TEXT,
    user TEXT NOT NULL DEFAULT 'dev',
    notes TEXT,
    created_at TEXT NOT NULL,
    superseded_by INTEGER REFERENCES pair_decisions(decision_id)
);
CREATE INDEX IF NOT EXISTS idx_pd_ab ON pair_decisions(ann_a_uuid, ann_b_uuid);
CREATE INDEX IF NOT EXISTS idx_pd_run ON pair_decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_pd_active ON pair_decisions(decision) WHERE superseded_by IS NULL;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    config_json TEXT NOT NULL,
    metrics_json TEXT,
    notes TEXT,
    git_sha TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('running','ok','failed')),
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_stage ON runs(stage);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS pair_queue (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    ann_a_uuid TEXT NOT NULL REFERENCES annotations(annotation_uuid),
    ann_b_uuid TEXT NOT NULL REFERENCES annotations(annotation_uuid),
    distance REAL NOT NULL,
    cluster_a INTEGER,
    cluster_b INTEGER,
    same_cluster INTEGER NOT NULL DEFAULT 0,
    position INTEGER NOT NULL,
    -- Pre-computed pair geometry (nullable: both annotations must have date/GPS)
    km_delta REAL,
    time_delta_days REAL,
    UNIQUE(run_id, ann_a_uuid, ann_b_uuid)
);
CREATE INDEX IF NOT EXISTS idx_pq_run_pos ON pair_queue(run_id, position);
