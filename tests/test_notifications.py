"""Actual SQL outbox, permissions, workflow routing and restart-safe in-app delivery."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta
import json
import pytest
from backend.core.database import transaction,utcnow,iso,audit
from backend.notifications.service import tick,inbox
from tests.test_phase4_copilot import copilot,ASSET
from tests.test_phase6_scheduling import planning,login,draft,APPROVAL
from tests.test_work_lifecycle import named_config,work,lifecycle
from tests.test_operations_coordination import coordinated,approved_schedule,handover


@pytest.fixture(autouse=True)
def isolate_workflow_events(request,monkeypatch):
    # Condition episodes are exercised separately with actual ingestion below.
    if 'critical_episode' not in request.node.name:monkeypatch.setattr('backend.notifications.conditions.scan',lambda conn,now:None)


def notices(c,kind=None,history=False):
    r=c.get('/api/notifications',params={'history':history});assert r.status_code==200,r.text
    return [n for n in r.json()['items'] if kind is None or n['kind']==kind]


def update(c,n,action='acknowledge',reason='I have received this and will review the linked evidence.'):
    return c.post('/api/notifications/'+n['id']+'/updates',json=dict(action=action,reason=reason))


def test_transactional_delivery_read_ack_and_approval_are_separate(planning):
    c=planning;wid,_=work(c,'draft');queued=notices(c,'work_review');assert len(queued)==1 and queued[0]['status']=='pending'
    assert update(c,queued[0]).status_code==409
    tick();n=notices(c,'work_review')[0];assert n['status']=='delivered'
    assert c.get('/api/notifications').json()['unread']==1
    assert update(c,n,'read').status_code==200
    assert c.get('/api/notifications').json()['unread']==0 and c.get('/api/notifications').json()['unacknowledged']==1
    assert update(c,n,reason='short').status_code==422
    assert update(c,n).status_code==200 and update(c,n).status_code==200
    assert not notices(c);h=notices(c,history=True)[0];assert h['acknowledged_at'] and h['acknowledgement']
    assert c.get('/api/work-orders').json()[0]['status']=='draft'
    with transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM audit_events WHERE action='notification_acknowledge'").fetchone()[0]==1


def test_concurrent_workers_deliver_once_and_restart_keeps_ack(planning):
    c=planning;work(c,'draft')
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda _:tick(),range(2)))
    n=notices(c)[0];assert n['attempts']==1;update(c,n);tick()
    assert not notices(c) and notices(c,history=True)[0]['acknowledged_at']


def test_domain_transaction_rollback_does_not_leave_notifications(planning):
    c=planning;wid,_=work(c,'draft')
    with transaction() as conn:before=conn.execute('SELECT COUNT(*) FROM notification_events').fetchone()[0]
    with pytest.raises(RuntimeError):
        with transaction() as conn:
            audit(conn,dict(id='manager',plant_id='demo-plant'),'draft_work_order',wid,'Rolled back domain event');raise RuntimeError('Injected transaction rollback')
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM notification_events').fetchone()[0]==before


@pytest.mark.parametrize('revocation',['asset','source','document','role','disabled'])
def test_revocation_hides_delivery_and_denies_ack(planning,revocation):
    c=planning;wid,_=work(c,'draft');tick();n=notices(c)[0]
    with transaction() as conn:
        if revocation=='asset':conn.execute('DELETE FROM asset_grants WHERE user_id=?',('manager',))
        elif revocation=='source':conn.execute('DELETE FROM source_grants WHERE user_id=?',('manager',))
        elif revocation=='document':conn.execute('DELETE FROM document_grants WHERE user_id=?',('manager',))
        elif revocation=='role':conn.execute("UPDATE users SET role='engineer' WHERE id='manager'")
        else:conn.execute("UPDATE users SET enabled=0 WHERE id='manager'")
    if revocation=='disabled':assert c.get('/api/notifications').status_code==401
    else:assert notices(c,history=True)==[]
    assert update(c,n).status_code in [401,404]


def test_inbox_owner_and_workspace_isolation(planning):
    c=planning;work(c,'draft');tick();n=notices(c)[0]
    c.headers['X-Maintenance-Workspace']='plant';assert notices(c,history=True)==[];assert update(c,n).status_code==404
    c.headers['X-Maintenance-Workspace']='demo';login(c,'engineer');assert notices(c)==[];assert update(c,n).status_code==404
    login(c,'manager');assert len(notices(c))==1


def test_assignment_change_routes_old_and_new_engineers_without_stale_queue(planning):
    c=planning;named_config(c);wid,p=work(c);tick()
    login(c,'engineer');assigned=notices(c,'work_assignment');assert len(assigned)==1
    login(c,'manager');assert lifecycle(c,wid,'replan').status_code==200;tick()
    login(c,'engineer');assert not notices(c,'work_assignment');changed=notices(c,'work_change');assert len(changed)==1
    assert update(c,assigned[0]).status_code==409
    assert update(c,changed[0]).status_code==200


def test_fresh_workflow_state_suppresses_obsolete_pending_review(planning):
    c=planning;wid,_=work(c,'draft');n=notices(c)[0]
    assert c.post('/api/work-orders/'+wid+'/review',json=APPROVAL).status_code==200;tick()
    assert notices(c,history=True)==[]
    with transaction() as conn:assert conn.execute('SELECT status FROM notification_deliveries WHERE id=?',(n['id'],)).fetchone()[0]=='suppressed'


def test_retry_backoff_failed_state_and_manual_recovery(planning,monkeypatch):
    c=planning;work(c,'draft');import backend.notifications.service as service
    real=service.deliver
    def fail(*args):raise RuntimeError('Injected in-app rendering failure, not external delivery')
    monkeypatch.setattr(service,'deliver',fail);now=utcnow()
    for attempt in range(5):tick(now+timedelta(hours=attempt))
    n=next(n for n in notices(c) if not n['escalated']);assert n['status']=='failed' and n['attempts']==5 and 'Injected' not in n['last_error']
    assert c.get('/api/notifications').json()['failed']==1
    monkeypatch.setattr(service,'deliver',real);assert update(c,n,'retry').status_code==200;tick()
    n=next(n for n in notices(c) if not n['escalated']);assert n['status']=='delivered' and n['attempts']==1


def test_escalates_once_not_on_read_and_ack_suppresses_later_escalation(planning):
    c=planning;wid,_=work(c,'draft');tick();n=notices(c)[0];update(c,n,'read')
    now=utcnow()+timedelta(minutes=61);tick(now);tick(now)
    all_items=notices(c);assert len([n for n in all_items if n['escalated']])==1
    tick(now+timedelta(hours=3));assert len([n for n in notices(c) if n['escalated']])==1
    assert update(c,n).status_code==200;assert not notices(c)
    assert len(notices(c,history=True))==2


def test_policy_version_and_disabled_escalation(planning):
    c=planning;body=dict(expected_version=0,escalation_minutes=1,escalation_enabled=False,reason='Disable escalation during the synthetic test rehearsal.')
    login(c,'manager');assert c.put('/api/notification-policy',json=body).status_code==403
    login(c,'admin');assert c.put('/api/notification-policy',json=body).status_code==200;assert c.put('/api/notification-policy',json=body).status_code==409
    wid,_=work(c,'draft');tick();tick(utcnow()+timedelta(days=2));assert not any(n['escalated'] for n in notices(c))


def test_operations_approval_release_and_return_routing(coordinated):
    c,wid,p,rid,uid=approved_schedule(coordinated);tick();login(c,'operations')
    ns=notices(c);assert any(n['kind']=='release_review' for n in ns)
    assert not any(n['kind']=='window_review' for n in ns)
    assert handover(c,wid,'release').status_code==200;tick();assert not notices(c,'release_review')
    from tests.test_field_work import change
    login(c,'engineer');change(c,wid,'start',site_clearance_confirmed=True);change(c,wid,'report_fixed',1);tick()
    login(c,'manager');assert len(notices(c,'repair_review'))==1;change(c,wid,'verify',2);tick()
    login(c,'operations');assert len(notices(c,'return_review'))==1
    assert handover(c,wid,'confirm_return',1).status_code==200;assert not notices(c,'return_review')


def test_critical_episode_dedup_stale_and_authoritative_recovery(planning):
    c=planning;login(c,'engineer')
    c.put('/api/simulation',json={'enabled':True,'scenario':'critical'})
    from backend.core.demo import simulate_tick
    simulate_tick();tick();critical=notices(c,'critical');assert critical
    original={n['id'] for n in critical};tick();assert {n['id'] for n in notices(c,'critical')}==original
    with transaction() as conn:
        for row in conn.execute('SELECT id,timestamp_utc FROM telemetry').fetchall():conn.execute('UPDATE telemetry SET timestamp_utc=? WHERE id=?',(iso(datetime.fromisoformat(row['timestamp_utc'])-timedelta(days=1)),row['id']))
    assert not notices(c,'critical');tick()
    simulate_tick();tick();assert {n['id'] for n in notices(c,'critical')}==original
    c.put('/api/simulation',json={'enabled':True,'scenario':'normal'});simulate_tick();tick();assert not notices(c,'critical')
    c.put('/api/simulation',json={'enabled':True,'scenario':'critical'});simulate_tick();tick();assert {n['id'] for n in notices(c,'critical')}.isdisjoint(original)


def test_issue_intake_assignment_escalation_and_deduplicated_request(planning):
    c=planning;login(c,'engineer');body=dict(asset_id=ASSET,title='Inspect reported bearing noise',description='Human report of unusual noise for maintenance review.',reported_severity='high',request_key='notification-issue-request-001')
    r=c.post('/api/issues',json=body);assert r.status_code==200;i=r.json();assert c.post('/api/issues',json=body).json()['id']==i['id'];tick()
    login(c,'manager');assert len(notices(c,'issue_review'))==1
    r=c.post('/api/issues/'+i['id']+'/updates',json=dict(action='assign',expected_version=i['version'],assignee_id='engineer',notes='Assign the reported observation for qualified investigation.'));assert r.status_code==200;tick();assert not notices(c,'issue_review')
    login(c,'engineer');n=notices(c,'issue_assignment')[0];assert n['status']=='delivered'
    tick(utcnow()+timedelta(minutes=61));tick(utcnow()+timedelta(minutes=61))
    login(c,'manager');assert len([n for n in notices(c,'issue_assignment') if n['escalated']])==1
    login(c,'engineer');update(c,n);login(c,'manager');assert not notices(c,'issue_assignment')


def test_round_assignment_and_revoked_multistep_evidence(planning):
    from tests.test_inspection_rounds import template,assign
    c=planning;t,_=template(c);r,body=assign(c,t);assert c.post('/api/inspections/templates/'+t['id']+'/assign',json=body).json()['id']==r['id'];tick()
    login(c,'engineer');n=notices(c,'round_assignment');assert len(n)==1 and n[0]['status']=='delivered'
    with transaction() as conn:conn.execute("DELETE FROM document_grants WHERE user_id='engineer'")
    assert notices(c,'round_assignment',history=True)==[];assert update(c,n[0]).status_code==404


def test_revoked_pending_delivery_is_suppressed_without_leaking_title(planning):
    c=planning;work(c,'draft');n=notices(c)[0]
    with transaction() as conn:conn.execute("DELETE FROM document_grants WHERE user_id='manager'")
    tick();assert not notices(c,history=True)
    with transaction() as conn:
        row=conn.execute('SELECT * FROM notification_deliveries WHERE id=?',(n['id'],)).fetchone();assert row['status']=='suppressed' and row['delivered_at'] is None


def test_event_replay_and_ack_race_are_idempotent(planning):
    from backend.notifications.events import capture
    from backend.notifications.service import update as apply
    from backend.routes.notifications import Update
    c=planning;wid,_=work(c,'draft')
    with transaction() as conn:
        event=conn.execute("SELECT id FROM audit_events WHERE action='draft_work_order' AND resource=?",(wid,)).fetchone()
        # Exact domain event replay cannot fan out duplicate messages.
        capture(conn,'draft_work_order',wid,event['id']);assert conn.execute("SELECT COUNT(*) FROM notification_events WHERE kind='work_review'").fetchone()[0]==1
    tick();n=notices(c)[0];user=dict(id='manager',role='maintenance_manager',plant_id='demo-plant');body=Update(action='acknowledge',reason='Acknowledged from concurrent test clients.')
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:apply(n['id'],body,user),range(2)))
    assert all(r['status']=='acknowledged' or r['status']=='acknowledge' for r in results)
    with transaction() as conn:assert conn.execute("SELECT COUNT(*) FROM audit_events WHERE action='notification_acknowledge'").fetchone()[0]==1
