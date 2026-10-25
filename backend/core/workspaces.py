"""Data-origin workspace views within an already authorized plant (not new tenants)."""
from fastapi import HTTPException


def asset_workspace(conn,asset_id):
    row=conn.execute("SELECT 1 FROM sensors s JOIN sources r ON r.id=s.source_id WHERE s.asset_id=? AND r.kind='simulation' LIMIT 1",(asset_id,)).fetchone()
    return 'demo' if row else 'plant'


def source_workspace(source):
    return 'demo' if source['kind']=='simulation' or source['id']=='demo-import' else 'plant'


def matches_asset(conn,user,asset_id):
    return not user.get('workspace') or asset_workspace(conn,asset_id)==user['workspace']


def matches_source(user,source):
    return not user.get('workspace') or source_workspace(source)==user['workspace']


def parse_workspace(value):
    if value is not None and value not in ('demo','plant'):raise HTTPException(422,'Workspace must be demo or plant')
    return value
