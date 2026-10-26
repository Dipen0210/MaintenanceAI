# Operations coordination

Open **Coordination** in the Operations workspace. This workflow records human decisions between maintenance and production; it does not operate equipment, issue site permits or establish physical machine health.

## Setup

1. An administrator creates or updates a user in **Administration** with the `operations_coordinator` role. Grant the relevant machines, data sources and approved documents. Demo accounts are not added automatically.
2. A maintenance manager or administrator chooses a machine in **Coordination → Machine coordination policy**, enables required operations approval and records a reason. Policy versions and reasons are retained. Configure each machine explicitly before proposing its downtime.
3. A manager reviews a work candidate through the existing maintenance workflow. Proposing its schedule creates an operations request for each required machine's exact proposed start/end window. Proposal generation reserves no resources.

Existing machines retain the prior maintenance-approval workflow until the policy is enabled. Changing policy is blocked while confirmed work remains open, including when its evidence permissions have subsequently been revoked. Pending requests are invalidated by a policy revision; create a new schedule proposal. A planning-resource revision also requires a new proposal.

## Downtime, repair and return

- An authorized operations coordinator reviews the work and references, then approves or declines the exact requested downtime window with a reason. A stale request or one whose window has passed cannot be approved.
- The maintenance manager confirms the schedule only after every required window is approved. Current resources, evidence access, coordinator eligibility and the exact approved windows are rechecked transactionally.
- Operations records **Release to maintenance** or **Hold for operations** before work starts. The engineer must still confirm site clearance, permits and isolation. Physical work must start within its confirmed window; only explicitly simulated work allows early rehearsal.
- The assigned engineer records work and reports the issue fixed. A different maintenance manager independently verifies the repair. For coordinated work, verification sets **Awaiting operations return**, not Completed.
- Operations independently confirms or declines return to production. Confirmation completes the work record. The return reviewer cannot be the repair reporter or verifier, even if that person's role has since changed. Decline keeps the job open.
- If further repair is needed, the manager can reopen the verified repair. This places the job back in Scheduled with an operations hold; a fresh release and engineer start are required. Physical start-window checks still apply.

Each mutation checks the latest version to prevent conflicting decisions. Current role, enabled identity, workspace, machine, source and approved-reference permissions are enforced by the backend. Disabled or revoked decision makers cannot authorize a later schedule confirmation or engineer start. An eligible coordinator can record a replacement decision with a new version and reason.

View open requests and handovers in **Coordination**; turn on historical decisions to see closed or superseded items. Work orders retain the operations history alongside field-work evidence. Copilot can cite the recorded handover state as historical workflow evidence; it cannot approve downtime, release equipment or confirm production return.

## Current boundaries

This feature is online and internal to PlantPilot. It does not send email/chat notifications, synchronize an ERP/MES, cancel or reschedule confirmed assignments, or implement a permit-to-work system. An expired physical window cannot be reopened: the original job stays awaiting return while a separately approved future corrective job is arranged. Do not interpret a UI record as authorization outside the approved window. Do not disable coordination on an open job to bypass a blocked step.

Local software tests and synthetic browser rehearsal are not factory commissioning or qualified expert approval. Expert Copilot review remains deferred.
