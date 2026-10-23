CREATE TABLE asset_profiles (id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id), config TEXT NOT NULL, actor_id TEXT NOT NULL REFERENCES users(id), timestamp_utc TEXT NOT NULL, active INTEGER NOT NULL CHECK(active IN(0,1)));
CREATE UNIQUE INDEX active_asset_profile ON asset_profiles(asset_id) WHERE active=1;
CREATE TABLE risk_policies (id TEXT PRIMARY KEY, plant_id TEXT NOT NULL REFERENCES plants(id), config TEXT NOT NULL, actor_id TEXT NOT NULL REFERENCES users(id), timestamp_utc TEXT NOT NULL, active INTEGER NOT NULL CHECK(active IN(0,1)));
CREATE UNIQUE INDEX active_risk_policy ON risk_policies(plant_id) WHERE active=1;
CREATE TABLE risk_assessments (id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id), policy_id TEXT NOT NULL REFERENCES risk_policies(id), profile_id TEXT NOT NULL REFERENCES asset_profiles(id), actor_id TEXT NOT NULL REFERENCES users(id), timestamp_utc TEXT NOT NULL, result TEXT NOT NULL);
CREATE INDEX risk_asset_time ON risk_assessments(asset_id,timestamp_utc);
