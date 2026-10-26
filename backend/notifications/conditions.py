"""One critical notification per episode; stale readings never imply recovery."""
from fastapi import HTTPException
from backend.core.security import authorize_asset,authorize_source
from backend.ingestion.condition import describe_asset
from backend.notifications.service import enqueue


def scan(conn,now):
    for asset in conn.execute('SELECT * FROM assets').fetchall():
        observation=None
        for row in conn.execute('SELECT id,role,plant_id FROM users WHERE plant_id=? AND enabled=1',(asset['plant_id'],)).fetchall():
            user=dict(row)
            try:
                authorize_asset(conn,user,asset['id'])
                for sensor in conn.execute('SELECT source_id FROM sensors WHERE asset_id=?',(asset['id'],)):authorize_source(conn,user,sensor['source_id'])
                observation=describe_asset(conn,user,dict(asset));break
            except HTTPException:continue
        if not observation:continue
        old=conn.execute('SELECT * FROM notification_conditions WHERE asset_id=?',(asset['id'],)).fetchone()
        if observation['condition']=='critical' and (not old or not old['is_open']):
            episode=old['episode']+1 if old else 1
            conn.execute('INSERT INTO notification_conditions VALUES(?,?,1,NULL) ON CONFLICT(asset_id) DO UPDATE SET episode=excluded.episode,is_open=1,event_id=NULL',(asset['id'],episode))
            eid=enqueue(conn,'critical',asset['id'],asset['id'],f"condition:{asset['id']}:{episode}",'critical',str(episode),now=now)
            conn.execute('UPDATE notification_conditions SET event_id=? WHERE asset_id=?',(eid,asset['id']))
        elif old and old['is_open'] and observation['condition']=='normal':
            conn.execute('UPDATE notification_conditions SET is_open=0 WHERE asset_id=?',(asset['id'],))
