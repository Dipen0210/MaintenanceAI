"""Transactional local SQL storage with ordered, versioned migrations."""
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import uuid

ROOT=Path(__file__).resolve().parents[2]


def utcnow():return datetime.now(timezone.utc)
def iso(value=None):return (value or utcnow()).isoformat()
def identifier():return str(uuid.uuid4())


def database_path():return Path(os.getenv('MAINTENANCEAI_DB',str(ROOT/'artifacts/maintenanceai.db')))


@contextmanager
def transaction():
    path=database_path();path.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(path,timeout=10)
    conn.row_factory=sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback();raise
    finally:conn.close()


def migrate():
    with transaction() as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('CREATE TABLE IF NOT EXISTS schema_versions(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)')
        for file in sorted((ROOT/'infrastructure/migrations').glob('*.sql')):
            if conn.execute('SELECT 1 FROM schema_versions WHERE version=?',(file.name,)).fetchone():continue
            # Individual SQL statements preserve atomic rollback of the migration.
            for statement in file.read_text().split(';'):
                if statement.strip():conn.execute(statement)
            conn.execute('INSERT INTO schema_versions VALUES(?,?)',(file.name,iso()))


def audit(conn, actor, action, resource, outcome):
    event_id=identifier()
    conn.execute('INSERT INTO audit_events VALUES(?,?,?,?,?,?,?)',
                 (event_id,actor.get('id') if actor else None,actor.get('plant_id') if actor else None,
                  action,resource,iso(),outcome))

    from backend.notifications.events import capture
    capture(conn,action,resource,event_id)
