# Notifications and acknowledgement

Open **Notifications** from the workspace navigation or the bell in the header. The badge counts unread, delivered messages that are still relevant and accessible to your account. Messages are scoped to the selected Demo or Plant view and your current machine, source and document grants.

## What appears

- Maintenance managers receive new work candidates, feasible schedule proposals, reported issues needing triage, repair reports awaiting verification and overdue acknowledgement escalations.
- Named engineers receive confirmed assignments, assigned issue investigations, inspection rounds and changes to their maintenance assignments.
- Operations coordinators receive proposed downtime windows, required equipment releases and production-return reviews.
- Authorized engineers, reliability engineers and maintenance managers receive critical sensor-limit observations. These are observations, not diagnoses or a declaration that a machine has failed.

The source workflow and notification outbox commit together. Repeating a successful idempotent issue/round request or replaying the same domain event does not create duplicate messages. Only events occurring after this feature is installed are routed; older work remains available in its existing queues. Critical readings already present when the worker starts can create the first condition episode.

## Read, acknowledge, act

**Mark read** or open the linked record to mark receipt as read. Reading does not stop escalation. Use **Acknowledge receipt**, enter a note, and submit to record your acknowledgement. This action does not approve a schedule, agree downtime, authorize maintenance, verify a repair, resolve an issue or clear a sensor alert. Perform those actions in the linked workflow.

The default inbox shows current unacknowledged messages, including pending/failed delivery. Enable **Show acknowledged and superseded messages** for accessible delivered history. Changed workflows stop contributing to current counts; queued messages that are obsolete or no longer authorized are suppressed. Revoked access hides titles, counts, links and acknowledgement actions immediately, including historical records. Former assignees retain a schedule-change message only while they retain the underlying evidence access.

## Escalation

An administrator can set the site acknowledgement deadline from 1 to 1,440 minutes; the default is 60 minutes. This policy applies to both workspace views. The deadline starts at the domain event's recorded creation time and is stored with that event. Deadline changes affect future events. Disabling escalation prevents additional escalation messages while preserving existing records.

The worker creates at most one escalation per original event, routed to currently authorized maintenance managers, if a delivered or terminally failed message remains unacknowledged and relevant. Reading alone does not stop escalation. Acknowledging the original message removes its escalation from the current queue once no original recipient still needs to acknowledge. Acknowledging an escalation records that manager's receipt; it does not impersonate the original recipients. Escalations do not escalate recursively.

Critical observations are deduplicated by machine episode. Fresh readings within all configured limits close the notification episode, without asserting physical health. Missing or stale readings hide the current critical message and pause escalation; they do not establish recovery or create a new episode. A later critical episode after a recorded within-limits observation produces a new message.

## Delivery and recovery

The application runs an in-app delivery worker every ten seconds. SQLite transactions serialize workers and retain outbox entries across restarts. Transient delivery failures retry after 30, 60, 120 and 240 seconds, up to five attempts. The inbox shows terminal failures; **Retry delivery** begins a new attempt cycle for a still-current message. Worker errors are logged, and the inbox reports a missing or older-than-one-minute heartbeat instead of presenting delivery as current.

This release supports **in-app delivery only**. No email, Slack or Teams messages are sent. An external channel, verified recipients, credentials and delivery policy must be selected and configured separately; this feature does not claim external delivery success. Offline/background push, quiet hours, multi-step escalation, incident ownership/suppression, enterprise delivery infrastructure and factory load validation remain future work. Notifications are not a safety alarm system; engineers must continue using the site's approved procedures and operational communication channels.
