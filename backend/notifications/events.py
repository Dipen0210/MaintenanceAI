"""Translate committed domain audit actions into outbox entries in the same transaction."""
import json
from backend.notifications.service import enqueue

ACTIONS={'draft_work_order','generate_preventive_draft','generate_meter_draft','review_schedule','propose_schedule',
 'field_work_report_fixed','field_work_verify','field_work_return_to_work','operations_release','operations_hold',
 'work_lifecycle_cancel','work_lifecycle_replan','report_issue','inspection_issue','issue_assign','assign_inspection_round'}


def capture(conn,action,rid,event_id):
    if action not in ACTIONS:return
    def emit(kind,resource,aid,audience,revision='',users=None):
        return enqueue(conn,kind,resource,aid,event_id+':'+kind+':'+resource,audience,str(revision),users)
    if action in ['report_issue','inspection_issue','issue_assign']:
        row=conn.execute('SELECT * FROM issues WHERE id=?',(rid,)).fetchone()
        if action=='issue_assign':emit('issue_assignment',rid,row['asset_id'],'engineer',row['assignee_id'],[row['assignee_id']])
        else:emit('issue_review',rid,row['asset_id'],'manager')
    elif action=='assign_inspection_round':
        row=conn.execute('SELECT * FROM inspection_runs WHERE id=?',(rid,)).fetchone();aid=json.loads(row['content'])['steps'][0]['asset_id']
        emit('round_assignment',rid,aid,'engineer',row['assignee_id'],[row['assignee_id']])
    elif action in ['propose_schedule','review_schedule']:
        p=conn.execute('SELECT * FROM schedule_proposals WHERE id=?',(rid,)).fetchone();result=json.loads(p['result'])
        if action=='propose_schedule' and result['feasibility']=='feasible':
            assignments=result['assignments']
            if assignments:emit('schedule_review',rid,assignments[0]['asset_id'],'manager')
            for row in conn.execute('SELECT * FROM operations_requests WHERE proposal_id=?',(rid,)).fetchall():
                w=conn.execute('SELECT asset_id FROM work_orders WHERE id=?',(row['work_order_id'],)).fetchone();emit('window_review',row['id'],w['asset_id'],'operations')
        elif action=='review_schedule' and p['status']=='approved':
            for a in result['assignments']:
                users=[r[0] for r in conn.execute('SELECT user_id FROM work_user_assignments WHERE work_order_id=?',(a['work_order_id'],))]
                emit('work_assignment',a['work_order_id'],a['asset_id'],'engineer',rid,users)
                if conn.execute('SELECT 1 FROM operations_handovers WHERE work_order_id=? AND status=\'awaiting_release\'',(a['work_order_id'],)).fetchone():emit('release_review',a['work_order_id'],a['asset_id'],'operations',rid)
    else:
        w=conn.execute('SELECT w.*,COALESCE(e.status,w.status) AS effective_status FROM work_orders w LEFT JOIN work_execution e ON e.work_order_id=w.id WHERE w.id=?',(rid,)).fetchone();aid=w['asset_id']
        if action in ['draft_work_order','generate_preventive_draft','generate_meter_draft']:emit('work_review',rid,aid,'manager')
        elif action=='field_work_report_fixed':emit('repair_review',rid,aid,'manager')
        elif action=='field_work_verify' and w['effective_status']=='awaiting_return_to_service':emit('return_review',rid,aid,'operations')
        elif action.startswith('work_lifecycle_'):
            old=conn.execute('SELECT snapshot FROM work_lifecycle_events WHERE work_order_id=? ORDER BY version DESC LIMIT 1',(rid,)).fetchone()
            users=[a['user_id'] for a in json.loads(old['snapshot'])['user_assignments']];emit('work_change',rid,aid,'engineer',users=users)
        elif action in ['operations_release','operations_hold','field_work_return_to_work']:
            users=[r[0] for r in conn.execute('SELECT user_id FROM work_user_assignments WHERE work_order_id=?',(rid,))];emit('work_change',rid,aid,'engineer',users=users)
