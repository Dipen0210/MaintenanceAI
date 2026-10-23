CREATE INDEX ingestion_events_asset_tag_time ON ingestion_events(asset_id,tag_id,timestamp_utc DESC);
