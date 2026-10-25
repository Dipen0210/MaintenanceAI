"""Backend-enforced identity and plant/asset/source authorization."""
from datetime import timedelta
import hashlib,hmac,secrets
from fastapi import Depends, HTTPException, Request
from backend.core.database import transaction,utcnow,iso,audit


def hash_password(password):
    salt=secrets.token_hex(16)
    key=hashlib.pbkdf2_hmac('sha256',password.encode(),salt.encode(),600000)
    return f'pbkdf2_sha256$600000${salt}${key.hex()}'


def verify_password(password,stored):
    try:
        _,rounds,salt,digest=stored.split('$')
        result=hashlib.pbkdf2_hmac('sha256',password.encode(),salt.encode(),int(rounds)).hex()
        return hmac.compare_digest(result,digest)
    except (ValueError,TypeError):return False


def authenticate(username,password):
    with transaction() as conn:
        user=conn.execute('SELECT * FROM users WHERE username=? AND enabled=1',(username,)).fetchone()
        if user is None or not verify_password(password,user['password_hash']):
            audit(conn,None,'login',username[:100],'denied')
            # Commit failed-attempt audit before returning an exception.
            conn.commit()
            raise HTTPException(401,'Invalid credentials')
        token=secrets.token_urlsafe(32)
        conn.execute('INSERT INTO sessions VALUES(?,?,?)',(hashlib.sha256(token.encode()).hexdigest(),user['id'],iso(utcnow()+timedelta(hours=8))))
        principal={k:user[k] for k in ['id','username','role','plant_id']}
        audit(conn,principal,'login',user['id'],'allowed')
        return token,principal


def principal(request:Request):
    authorization=request.headers.get('authorization','')
    if authorization and not authorization.startswith('Bearer '):raise HTTPException(401,'Unsupported authorization scheme')
    token=authorization[7:] if authorization.startswith('Bearer ') else request.cookies.get('maintenanceai_session')
    if not token:raise HTTPException(401,'Authentication required')
    # Browser cookie mutations require an approved same-site origin. CLI tools use Bearer tokens.
    if not authorization and request.method not in ['GET','HEAD','OPTIONS']:
        if request.headers.get('origin') not in ['http://localhost:3000','http://127.0.0.1:3000']:
            raise HTTPException(403,'Unapproved request origin')
    with transaction() as conn:
        user=conn.execute('''SELECT u.id,u.username,u.role,u.plant_id FROM users u
            JOIN sessions s ON s.user_id=u.id WHERE s.token_hash=? AND s.expires_at>? AND u.enabled=1''',
            (hashlib.sha256(token.encode()).hexdigest(),iso())).fetchone()
        if not user:raise HTTPException(401,'Session expired or revoked')
        result=dict(user)
        from backend.core.workspaces import parse_workspace
        workspace=parse_workspace(request.headers.get('x-maintenance-workspace'))
        # Explicit site administration/resource configuration retains whole-site scope.
        if workspace and not request.url.path.startswith(('/api/admin/','/api/planning/config')):result['workspace']=workspace
        return result


def require_roles(*roles):
    def check(user=Depends(principal)):
        if user['role'] not in roles:raise HTTPException(403,'Role does not permit this action')
        return user
    return check


def allowed_assets(conn,user):
    from backend.core.workspaces import matches_asset
    return [dict(r) for r in conn.execute('''SELECT a.* FROM assets a JOIN asset_grants g
        ON a.id=g.asset_id WHERE g.user_id=? AND a.plant_id=? ORDER BY a.id''',(user['id'],user['plant_id'])) if matches_asset(conn,user,r['id'])]


def authorize_asset(conn,user,asset_id):
    row=conn.execute('''SELECT a.* FROM assets a JOIN asset_grants g ON a.id=g.asset_id
        WHERE a.id=? AND g.user_id=? AND a.plant_id=?''',(asset_id,user['id'],user['plant_id'])).fetchone()
    from backend.core.workspaces import matches_asset
    if row is None or not matches_asset(conn,user,asset_id):raise HTTPException(404,'Asset not found or not authorized in this workspace')
    return dict(row)


def authorize_source(conn,user,source_id):
    row=conn.execute('''SELECT s.* FROM sources s JOIN source_grants g ON s.id=g.source_id
        WHERE s.id=? AND g.user_id=? AND s.plant_id=?''',(source_id,user['id'],user['plant_id'])).fetchone()
    from backend.core.workspaces import matches_source
    if row is None or not matches_source(user,row):raise HTTPException(404,'Source not found or not authorized in this workspace')
    return dict(row)


def allowed_source_ids(conn,user):
    from backend.core.workspaces import matches_source
    return [r['id'] for r in conn.execute('SELECT s.* FROM sources s JOIN source_grants g ON g.source_id=s.id WHERE g.user_id=? AND s.plant_id=?',(user['id'],user['plant_id'])) if matches_source(user,r)]
