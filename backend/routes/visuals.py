"""Authorized equipment photos and user-positioned sensor maps."""
import io,json,warnings
from fastapi import APIRouter,Depends,File,HTTPException,Response,UploadFile
from pydantic import Field
from PIL import Image,ImageOps,UnidentifiedImageError
from backend.core.database import transaction,iso,audit
from backend.core.security import principal,authorize_asset,authorize_source
from backend.routes.administration import admin,scope
from backend.scheduling.contracts import Strict

router=APIRouter(prefix='/api',tags=['equipment sensor map'])


class Marker(Strict):
    tag_id:str=Field(min_length=1,max_length=100)
    label:str=Field(min_length=3,max_length=120)
    x:float=Field(ge=0,le=100,allow_inf_nan=False)
    y:float=Field(ge=0,le=100,allow_inf_nan=False)


class Layout(Strict):
    markers:list[Marker]=Field(min_length=1,max_length=20)


def view(conn,user,asset_id):
    authorize_asset(conn,user,asset_id)
    sensors=[dict(r) for r in conn.execute('SELECT s.* FROM sensors s JOIN source_grants g ON g.source_id=s.source_id WHERE s.asset_id=? AND g.user_id=? ORDER BY s.tag_id',(asset_id,user['id']))]
    row=conn.execute('SELECT markers,photo IS NOT NULL AS has_photo FROM asset_visuals WHERE asset_id=?',(asset_id,)).fetchone()
    stored=json.loads(row['markers']) if row and row['markers'] else []
    defaults={'current':(24,28,'Electrical panel / drive'),'temperature':(40,41,'Motor / bearing housing'),'vibration_rms':(61,54,'Bearing housing')}
    markers=[]
    for i,s in enumerate(sensors):
        marker=next((m for m in stored if m['tag_id']==s['tag_id']),None)
        if marker is None:
            x,y,label=defaults.get(s['tag_id'],(28+(i%4)*16,38+(i//4)*15,s['tag_id'].replace('_',' ')))
            marker=dict(tag_id=s['tag_id'],label=label,x=x,y=y)
        markers.append(marker)
    return dict(asset_id=asset_id,markers=markers,layout_kind='configured' if stored else 'illustrative',photo_available=bool(row and row['has_photo']))


@router.get('/assets/{asset_id}/diagram')
def diagram(asset_id:str,user=Depends(principal)):
    with transaction() as conn:return view(conn,user,asset_id)


@router.put('/admin/assets/{asset_id}/diagram')
def save(asset_id:str,body:Layout,user=Depends(admin)):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope(conn,user);authorize_asset(conn,user,asset_id)
        sensors={r['tag_id']:r['source_id'] for r in conn.execute('SELECT tag_id,source_id FROM sensors WHERE asset_id=?',(asset_id,))}
        if len({m.tag_id for m in body.markers})!=len(body.markers) or {m.tag_id for m in body.markers}!=set(sensors):raise HTTPException(422,'Place each configured sensor exactly once')
        for sid in set(sensors.values()):authorize_source(conn,user,sid)
        conn.execute('INSERT INTO asset_visuals(asset_id,markers,updated_at,actor_id) VALUES(?,?,?,?) ON CONFLICT(asset_id) DO UPDATE SET markers=excluded.markers,updated_at=excluded.updated_at,actor_id=excluded.actor_id',(asset_id,json.dumps([m.model_dump() for m in body.markers]),iso(),user['id']))
        audit(conn,user,'configure_sensor_locations',asset_id,'user-positioned visual map; not an installation certification')
        return view(conn,user,asset_id)


@router.get('/assets/{asset_id}/photo')
def photo(asset_id:str,user=Depends(principal)):
    with transaction() as conn:
        authorize_asset(conn,user,asset_id)
        row=conn.execute('SELECT photo,mime_type FROM asset_visuals WHERE asset_id=?',(asset_id,)).fetchone()
        if not row or not row['photo']:raise HTTPException(404,'No machine photo uploaded')
        return Response(row['photo'],media_type=row['mime_type'],headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})


@router.put('/admin/assets/{asset_id}/photo')
def upload(asset_id:str,file:UploadFile=File(...),user=Depends(admin)):
    with transaction() as conn:scope(conn,user);authorize_asset(conn,user,asset_id)
    content=file.file.read(5*1024*1024+1)
    if len(content)>5*1024*1024:raise HTTPException(413,'Machine photo must be under 5 MiB')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in ['PNG','JPEG'] or image.width*image.height>12000000:raise ValueError('Expected a JPEG or PNG of at most 12 megapixels')
                image=ImageOps.exif_transpose(image).convert('RGB');image.thumbnail((2000,1400))
                output=io.BytesIO();image.save(output,format='JPEG',quality=90)
    except (ValueError,OSError,UnidentifiedImageError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
        raise HTTPException(422,'Use a valid JPEG or PNG photo of at most 12 megapixels') from exc
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope(conn,user);authorize_asset(conn,user,asset_id)
        conn.execute('INSERT INTO asset_visuals(asset_id,markers,photo,mime_type,updated_at,actor_id) VALUES(?,NULL,?,?,?,?) ON CONFLICT(asset_id) DO UPDATE SET markers=NULL,photo=excluded.photo,mime_type=excluded.mime_type,updated_at=excluded.updated_at,actor_id=excluded.actor_id',(asset_id,output.getvalue(),'image/jpeg',iso(),user['id']))
        audit(conn,user,'upload_machine_photo',asset_id,'photo normalized; position sensor markers on the new photo')
        return view(conn,user,asset_id)
