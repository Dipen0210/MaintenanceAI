CREATE TABLE model_versions (id TEXT PRIMARY KEY, family TEXT NOT NULL, version TEXT NOT NULL, kind TEXT NOT NULL, metadata TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0 CHECK(active IN (0,1)));
CREATE UNIQUE INDEX active_model_family ON model_versions(family) WHERE active=1;
CREATE TABLE model_bindings (asset_id TEXT NOT NULL REFERENCES assets(id), family TEXT NOT NULL, dataset_id TEXT NOT NULL, PRIMARY KEY(asset_id,family));
CREATE TABLE model_activations (id TEXT PRIMARY KEY, family TEXT NOT NULL, previous_id TEXT, next_id TEXT NOT NULL REFERENCES model_versions(id), actor_id TEXT NOT NULL REFERENCES users(id), timestamp_utc TEXT NOT NULL, reason TEXT NOT NULL);
CREATE TABLE signal_inputs (id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id), source_id TEXT NOT NULL REFERENCES sources(id), kind TEXT NOT NULL, timestamp_utc TEXT NOT NULL, metadata TEXT NOT NULL, content_path TEXT NOT NULL, sha256 TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE predictions (id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id), model_id TEXT NOT NULL REFERENCES model_versions(id), input_id TEXT REFERENCES signal_inputs(id), timestamp_utc TEXT NOT NULL, status TEXT NOT NULL, result TEXT NOT NULL, latency_ms REAL NOT NULL, actor_id TEXT NOT NULL REFERENCES users(id));
CREATE INDEX prediction_asset_time ON predictions(asset_id,timestamp_utc);
