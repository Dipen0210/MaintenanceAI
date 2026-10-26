"""Reviewed part identity and replenishment signals, independent of fabricated forecasts."""
import json
from fastapi import HTTPException
from backend.core.database import transaction,identifier,iso,audit
from backend.core.security import authorize_asset,allowed_assets
from backend.core.workspaces import asset_workspace
from backend.inventory.service import scope_access,enabled,part,catalog as stock_catalog

DEFAULT=dict(manufacturer='',manufacturer_part_number='',unit='units',location='',minimum_stock=0,target_stock=0,lead_time_days=0,compatible_assets=[],substitute_ids=[])


def details(conn,plant,scope,pid):
    row=conn.execute('SELECT * FROM inventory_part_details WHERE plant_id=? AND workspace=? AND part_id=?',(plant,scope,pid)).fetchone()
    return dict(content=json.loads(row['content']) if row else dict(DEFAULT),version=row['version'] if row else 0,updated_by=row['updated_by'] if row else None,updated_at=row['updated_at'] if row else None)


def listing(user):
    from backend.scheduling.service import list_orders
    data=stock_catalog(user)
    with transaction() as conn:
        scope=scope_access(conn,user);demand={}
        for w in list_orders(conn,user,limit=1000):
            if w['status']=='approved':
                for pid,quantity in w['content']['parts'].items():demand[pid]=demand.get(pid,0)+quantity
        for p in data['parts']:
            d=details(conn,user['plant_id'],scope,p['part_id']);p['details']=d
            low=p['available']<=d['content']['minimum_stock'];shortage=max(0,demand.get(p['part_id'],0)-p['available'])
            p['replenishment']=dict(low_stock=low,approved_demand=demand.get(p['part_id'],0),shortage=shortage,suggested_quantity=max(shortage,d['content']['target_stock']-p['available'] if low else 0),basis='Configured min/target and accessible approved unscheduled work; not a demand forecast')
    return data


def save(pid,body,user,create=False):
    from backend.scheduling.service import active_config
    from backend.scheduling.contracts import PlanningConfig
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope=scope_access(conn,user,['maintenance_manager','administrator'])
        if not enabled(conn,user['plant_id'],scope):raise HTTPException(409,'Activate counted inventory first')
        if create:
            if conn.execute('SELECT 1 FROM inventory_parts WHERE plant_id=? AND workspace=? AND part_id=?',(user['plant_id'],scope,pid)).fetchone():raise HTTPException(409,'Part identifier already exists; edit the existing catalog entry')
            whole={**user};whole.pop('workspace',None)
            all_ids={r[0] for r in conn.execute('SELECT id FROM assets WHERE plant_id=?',(user['plant_id'],))}
            if all_ids!={a['id'] for a in allowed_assets(conn,whole)}:raise HTTPException(403,'Adding a planning part requires whole-plant asset access')
            cfg=active_config(conn,whole)['config']
            prior=next((p for p in cfg['parts'] if p['id']==pid),None)
            if prior and prior['name']!=body.name:raise HTTPException(409,'Part identifier belongs to a differently named planning resource')
            if not prior:
                if len(cfg['parts'])>=100:raise HTTPException(422,'Planning currently supports at most 100 part identifiers')
                cfg['parts'].append(dict(id=pid,name=body.name,quantity=0,available_at=iso()));cfg['reason']=body.reason
                validated=PlanningConfig.model_validate(cfg).model_dump(mode='json')
                conn.execute('UPDATE planning_configs SET active=0 WHERE plant_id=?',(user['plant_id'],))
                conn.execute('INSERT INTO planning_configs VALUES(?,?,?,?,?,1)',(identifier(),user['plant_id'],json.dumps(validated),user['id'],iso()))
                from backend.inventory.service import sync_catalog
                for s in ['demo','plant']:
                    if enabled(conn,user['plant_id'],s):sync_catalog(conn,user['plant_id'],s,validated['parts'])
            else:conn.execute('INSERT INTO inventory_parts(plant_id,workspace,part_id,name) VALUES(?,?,?,?)',(user['plant_id'],scope,pid,body.name))
        p=part(conn,user['plant_id'],scope,pid);old=details(conn,user['plant_id'],scope,pid)
        if old['version']!=body.expected_version:raise HTTPException(409,'Part details changed; refresh before editing')
        content=body.model_dump(exclude={'expected_version','reason','name','part_id'})
        if content['unit']!=old['content']['unit'] and conn.execute('SELECT 1 FROM inventory_movements WHERE plant_id=? AND workspace=? AND part_id=?',(user['plant_id'],scope,pid)).fetchone():raise HTTPException(409,'Units are fixed after stock movements; use a distinct part identifier for a different unit')
        for aid in content['compatible_assets']:
            authorize_asset(conn,user,aid)
            if asset_workspace(conn,aid)!=scope:raise HTTPException(422,'Compatible machines must belong to the same workspace')
        for sid in content['substitute_ids']:
            if sid==pid:raise HTTPException(422,'A part cannot substitute itself')
            part(conn,user['plant_id'],scope,sid)
            if details(conn,user['plant_id'],scope,sid)['content']['unit']!=content['unit']:raise HTTPException(422,'Reviewed alternatives must use the same stock unit')
        # Existing incoming alternative references also constrain unit changes.
        for row in conn.execute('SELECT content FROM inventory_part_details WHERE plant_id=? AND workspace=?',(user['plant_id'],scope)):
            other=json.loads(row[0])
            if pid in other['substitute_ids'] and other['unit']!=content['unit']:raise HTTPException(409,'Remove incompatible alternative references before changing units')
        now=iso();conn.execute('INSERT INTO inventory_part_details VALUES(?,?,?,?,?,?,?) ON CONFLICT(plant_id,workspace,part_id) DO UPDATE SET content=excluded.content,version=excluded.version,updated_by=excluded.updated_by,updated_at=excluded.updated_at',(user['plant_id'],scope,pid,json.dumps(content),old['version']+1,user['id'],now))
        conn.execute('INSERT INTO inventory_catalog_events VALUES(?,?,?,?,?,?,?,?)',(identifier(),user['plant_id'],scope,pid,user['id'],now,body.reason,json.dumps(content)))
        audit(conn,user,'review_part_catalog',pid,body.reason)
        return dict(part_id=p['part_id'],version=old['version']+1)
