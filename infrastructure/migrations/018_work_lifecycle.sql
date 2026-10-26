CREATE TABLE schedule_assignment_archive(id TEXT PRIMARY KEY,proposal_id TEXT NOT NULL REFERENCES schedule_proposals(id),work_order_id TEXT NOT NULL REFERENCES work_orders(id),asset_id TEXT NOT NULL REFERENCES assets(id),start_utc TEXT NOT NULL,end_utc TEXT NOT NULL,technician_ids TEXT NOT NULL,parts TEXT NOT NULL,disposition TEXT NOT NULL,actor_id TEXT NOT NULL REFERENCES users(id),changed_at TEXT NOT NULL,reason TEXT NOT NULL);
CREATE INDEX archived_assignments_work ON schedule_assignment_archive(work_order_id,changed_at);
CREATE TABLE work_lifecycle_events(id TEXT PRIMARY KEY,work_order_id TEXT NOT NULL REFERENCES work_orders(id),version INTEGER NOT NULL,actor_id TEXT NOT NULL REFERENCES users(id),created_at TEXT NOT NULL,action TEXT NOT NULL,reason TEXT NOT NULL,snapshot TEXT NOT NULL,UNIQUE(work_order_id,version));
CREATE TABLE work_execution_next(work_order_id TEXT PRIMARY KEY REFERENCES work_orders(id),status TEXT NOT NULL CHECK(status IN('approved','scheduled','in_progress','awaiting_verification','awaiting_return_to_service','completed','cancelled')),version INTEGER NOT NULL,updated_at TEXT NOT NULL);
INSERT INTO work_execution_next SELECT * FROM work_execution;
DROP TABLE work_execution;
ALTER TABLE work_execution_next RENAME TO work_execution;
