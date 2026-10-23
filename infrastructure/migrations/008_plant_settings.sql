CREATE TABLE plant_settings(plant_id TEXT PRIMARY KEY REFERENCES plants(id), location TEXT NOT NULL, configuration TEXT NOT NULL, updated_at TEXT NOT NULL, actor_id TEXT REFERENCES users(id));
INSERT INTO plant_settings(plant_id,location,configuration,updated_at,actor_id) SELECT id,'Unconfigured','{"display_timezone":"UTC"}',strftime('%Y-%m-%dT%H:%M:%SZ','now'),NULL FROM plants;
