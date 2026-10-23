CREATE TABLE asset_visuals(asset_id TEXT PRIMARY KEY REFERENCES assets(id), markers TEXT, photo BLOB, mime_type TEXT, updated_at TEXT NOT NULL, actor_id TEXT NOT NULL REFERENCES users(id));
