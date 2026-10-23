ALTER TABLE simulation_controls ADD COLUMN scenario TEXT NOT NULL DEFAULT 'mixed' CHECK(scenario IN('mixed','normal','warning','critical'));
CREATE TABLE work_execution(work_order_id TEXT PRIMARY KEY REFERENCES work_orders(id),status TEXT NOT NULL CHECK(status IN('scheduled','in_progress','awaiting_verification','completed')),version INTEGER NOT NULL,updated_at TEXT NOT NULL);
CREATE TABLE work_events(id TEXT PRIMARY KEY,work_order_id TEXT NOT NULL REFERENCES work_orders(id),actor_id TEXT NOT NULL REFERENCES users(id),created_at TEXT NOT NULL,action TEXT NOT NULL,status TEXT NOT NULL,notes TEXT NOT NULL,version INTEGER NOT NULL,UNIQUE(work_order_id,version));
CREATE INDEX work_events_order ON work_events(work_order_id,version);
