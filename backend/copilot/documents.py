"""Approved, equipment-scoped document storage and real neural vector retrieval."""
import io,json,re,sqlite3
import numpy as np
from fastapi import HTTPException
from backend.core.database import transaction,identifier,iso,database_path,audit
from backend.core.security import authorize_asset,allowed_assets
from backend.copilot.providers import embed
from backend.copilot.safety import quarantine_reason
from backend.copilot.configuration import RETRIEVAL


def extract(content,filename):
    if filename.lower().endswith('.pdf'):
        if any(marker in content for marker in [b'/JavaScript',b'/Launch',b'/EmbeddedFile']):raise ValueError('Active or embedded PDF content is not accepted')
        import pdfplumber
        pages=[]
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            if len(pdf.pages)>100:raise ValueError('Document exceeds 100 pages')
            for i,page in enumerate(pdf.pages):
                text=page.extract_text() or ''
                tables=page.extract_tables()
                for table in tables:
                    text+='\n\n'+ '\n'.join(' | '.join(str(cell or '') for cell in row) for row in table)
                pages.append((i+1,text))
    elif filename.lower().endswith(('.txt','.md')):
        text=content.decode('utf-8')
        pages=[(i+1,page) for i,page in enumerate(text.split('\f'))]
    else:raise ValueError('Supported documents are PDF, UTF-8 Markdown or text')
    if sum(len(text) for _,text in pages)>1000000:raise ValueError('Extracted text exceeds one million characters')
    if not any(text.strip() for _,text in pages):raise ValueError('No readable text; scanned documents require a reviewed OCR import')
    chunks=[]
    for page,text in pages:
        section='Page '+str(page)
        for paragraph in re.split(r'\n\s*\n',text):
            lines=paragraph.strip().splitlines()
            if not lines:continue
            if lines[0].startswith('#'):
                section=lines[0].lstrip('# ').strip();paragraph='\n'.join(lines[1:]).strip()
            if not paragraph:continue
            # Keep chunks inside the MiniLM 256-wordpiece context with conservative word windows.
            words=paragraph.split()
            for start in range(0,len(words),RETRIEVAL['chunk_stride']):
                text_chunk=' '.join(words[start:start+RETRIEVAL['chunk_words']])
                if text_chunk:chunks.append(dict(id=identifier(),page=page,section=section,content=text_chunk))
    if not chunks or len(chunks)>1000:raise ValueError('Expected 1–1000 text chunks')
    return chunks


def authorize_document(conn,user,document_id,asset_id=None,require_approved=True):
    row=conn.execute('''SELECT d.* FROM documents d JOIN document_grants g ON d.id=g.document_id
                       WHERE d.id=? AND g.user_id=? AND d.plant_id=?''',(document_id,user['id'],user['plant_id'])).fetchone()
    if not row or (require_approved and row['status']!='approved'):raise HTTPException(404,'Document not found or not authorized')
    doc=dict(row)
    if asset_id:
        asset=authorize_asset(conn,user,asset_id)
        if asset['equipment_type'] not in json.loads(doc['equipment_types']):raise HTTPException(404,'Document is not applicable to this equipment')
        scoped=json.loads(doc['asset_ids'])
        if scoped and asset_id not in scoped:raise HTTPException(404,'Document is not applicable to this asset')
        if json.loads(doc['metadata']).get('simulation_only'):
            mode=conn.execute('SELECT mode FROM plants WHERE id=?',(asset['plant_id'],)).fetchone()[0]
            from backend.core.workspaces import asset_workspace
            if mode!='simulation' or asset_workspace(conn,asset_id)!='demo':raise HTTPException(404,'Simulation documentation is unavailable for physical equipment')
    return doc


def import_document(content,filename,metadata,user):
    import hashlib
    if not content or len(content)>20*1024*1024:raise HTTPException(413,'Document must contain 1 byte to 20 MiB')
    with transaction() as conn:
        for asset_id in metadata.asset_ids:
            asset=authorize_asset(conn,user,asset_id)
            if asset['equipment_type'] not in metadata.equipment_types:raise HTTPException(422,'Asset type does not match document applicability')
        for uid in metadata.reader_ids:
            if not conn.execute('SELECT 1 FROM users WHERE id=? AND plant_id=? AND enabled=1',(uid,user['plant_id'])).fetchone():raise HTTPException(422,'Document reader is not an enabled user in this plant')
    try:chunks=extract(content,filename)
    except (ValueError,UnicodeError) as exc:raise HTTPException(422,str(exc)) from exc
    except Exception as exc:raise HTTPException(422,'Document extraction failed') from exc
    reason=quarantine_reason('\n'.join(c['content'] for c in chunks))
    if reason:vectors=None;version=None
    else:
        try:
            batches=[];version=None
            for start in range(0,len(chunks),32):
                vectors,version=embed([c['section']+' '+c['content'] for c in chunks[start:start+32]]);batches.append(vectors)
            vectors=np.concatenate(batches)
        except (OSError,ValueError,ImportError) as exc:raise HTTPException(503,'Verified embedding model unavailable') from exc
    doc_id=identifier();path=database_path().parent/'documents'/doc_id;path.parent.mkdir(parents=True,exist_ok=True)
    info=dict(simulation_only=metadata.simulation_only,original_filename=filename,extraction_version='page-section-table-v1',quarantine_reason=reason)
    stamp=iso();digest=hashlib.sha256(content).hexdigest();path.write_bytes(content)
    try:
        with transaction() as conn:
            conn.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                         (doc_id,user['plant_id'],metadata.title,metadata.revision,json.dumps(metadata.equipment_types),json.dumps(metadata.asset_ids),
                          'quarantined' if reason else 'pending',str(path),digest,user['id'],stamp,None,None,json.dumps(info)))
            for uid in set(metadata.reader_ids+[user['id']]):conn.execute('INSERT INTO document_grants VALUES(?,?)',(uid,doc_id))
            for i,c in enumerate(chunks):
                conn.execute('INSERT INTO document_chunks VALUES(?,?,?,?,?,?,?,?,0)',
                             (c['id'],doc_id,c['page'],c['section'],c['content'],'reference',vectors[i].astype('<f4').tobytes() if vectors is not None else None,version))
            audit(conn,user,'import_document',doc_id,'quarantined' if reason else 'pending review')
    except sqlite3.IntegrityError as exc:path.unlink(missing_ok=True);raise HTTPException(409,'Document title/revision already exists') from exc
    except BaseException:path.unlink(missing_ok=True);raise
    return {'id':doc_id,'title':metadata.title,'revision':metadata.revision,'status':'quarantined' if reason else 'pending','chunk_count':len(chunks),'quarantine_reason':reason}


def authorize_document_review(conn,user,document_id):
    doc=authorize_document(conn,user,document_id,require_approved=False)
    for aid in json.loads(doc['asset_ids']):authorize_asset(conn,user,aid)
    if not json.loads(doc['asset_ids']) and not any(a['equipment_type'] in json.loads(doc['equipment_types']) for a in allowed_assets(conn,user)):
        raise HTTPException(404,'Document scope is not authorized')
    return doc


def approve(document_id,body,user):
    with transaction() as conn:
        doc=authorize_document_review(conn,user,document_id)
        if doc['status']!='pending':raise HTTPException(409,'Only pending, non-quarantined revisions may be approved')
        chunks={r['id']:dict(r) for r in conn.execute('SELECT * FROM document_chunks WHERE document_id=?',(document_id,))}
        if len({c.id for c in body.chunks})!=len(body.chunks):raise HTTPException(422,'Duplicate reviewed chunk')
        for reviewed in body.chunks:
            if reviewed.id not in chunks:raise HTTPException(422,'Reviewed chunk is outside this document')
            if quarantine_reason(chunks[reviewed.id]['content']):raise HTTPException(422,'Unsafe embedded content cannot be approved')
        for reviewed in body.chunks:conn.execute('UPDATE document_chunks SET reviewed=1,kind=? WHERE id=?',(reviewed.kind,reviewed.id))
        metadata=json.loads(doc['metadata']);metadata['chunk_fault_classes']={c.id:c.fault_classes for c in body.chunks if c.kind=='hypothesis'}
        conn.execute("UPDATE documents SET status='approved',approved_by=?,approved_at=?,metadata=? WHERE id=?",(user['id'],iso(),json.dumps(metadata),document_id))
        audit(conn,user,'approve_document',document_id,body.reason)
        return {'id':document_id,'status':'approved','reviewed_chunks':len(body.chunks)}


def search(user,asset_id,query,limit=5):
    # Filter access/applicability before computing similarities or returning text.
    with transaction() as conn:
        authorize_asset(conn,user,asset_id)
        ids=[r['id'] for r in conn.execute('''SELECT d.id FROM documents d JOIN document_grants g ON d.id=g.document_id
                     WHERE g.user_id=? AND d.plant_id=? AND d.status='approved' ''',(user['id'],user['plant_id']))]
        chunks=[]
        for did in ids:
            try:doc=authorize_document(conn,user,did,asset_id)
            except HTTPException:continue
            for row in conn.execute('SELECT * FROM document_chunks WHERE document_id=? AND reviewed=1',(did,)):
                chunks.append({**dict(row),'title':doc['title'],'revision':doc['revision']})
        audit(conn,user,'retrieve_manual',asset_id,'authorized candidates '+str(len(chunks)))
    if not chunks:return []
    vector,version=embed([query]);ranked=[]
    for c in chunks:
        if c['embedding_version']!=version:continue
        stored=np.frombuffer(c['embedding'],dtype='<f4')
        if stored.shape!=(384,) or not np.isfinite(stored).all():continue
        score=float(vector[0]@stored)
        if score<RETRIEVAL['minimum_similarity']:continue
        ranked.append({'id':c['id'],'document_id':c['document_id'],'title':c['title'],'revision':c['revision'],
                       'page':c['page'],'section':c['section'],'content':c['content'],'kind':c['kind'],'score':score,'embedding_version':version})
    return sorted(ranked,key=lambda r:(-r['score'],r['id']))[:limit]
