"""QR links identify equipment; they never confer permission or contain credentials."""
import io,os
from urllib.parse import urlencode,urlsplit
from fastapi import APIRouter,Depends,HTTPException,Response
import qrcode
from backend.core.database import transaction
from backend.core.security import principal,authorize_asset
from backend.core.workspaces import asset_workspace

router=APIRouter(prefix='/api',tags=['asset quick access'])

@router.get('/asset-access/{aid}')
def resolve(aid:str,user=Depends(principal)):
    # Resolve data origin across views, retaining all actual plant/asset grants.
    with transaction() as conn:
        authorize_asset(conn,{**user,'workspace':None},aid)
        return {'asset_id':aid,'workspace':asset_workspace(conn,aid)}

@router.get('/assets/{aid}/qr')
def label(aid:str,origin:str,user=Depends(principal)):
    allowed={'http://localhost:3000','http://127.0.0.1:3000'}
    configured=os.getenv('MAINTENANCEAI_PUBLIC_WEB_ORIGIN')
    if configured:allowed.add(configured.rstrip('/'))
    if origin not in allowed:raise HTTPException(422,'QR labels require an approved application origin')
    try:parsed=urlsplit(origin)
    except ValueError:raise HTTPException(422,'QR labels require an approved application origin')
    if origin not in allowed or parsed.scheme not in ('http','https') or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
        raise HTTPException(422,'QR labels require an approved application origin')
    with transaction() as conn:authorize_asset(conn,user,aid)
    payload=origin+'/operations/scan?'+urlencode({'asset':aid})
    qr=qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M,box_size=8,border=4)
    qr.add_data(payload);qr.make(fit=True);image=qr.make_image(fill_color='black',back_color='white')
    out=io.BytesIO();image.save(out,format='PNG')
    return Response(out.getvalue(),media_type='image/png',headers={'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'})
