# Field work, team handover and repair verification

The workspace records the maintenance team's decisions. It does not control machines, issue site permits, automatically clear alarms, or independently prove a repair succeeded.

## Try warning and critical conditions

In the workspace header, enable **Run simulation** and select a **Demo condition scenario**:

- **Mixed conditions**: `PLA-PUMP-002` has critical synthetic readings, the demo conveyor has warning readings, and other demo assets remain within their default limits.
- **All within limits**: demo values return to their normal range.
- **Conveyor warning** or **Conveyor critical**: investigate a controlled incident on `DEMO-CONVEYOR-001`.

Conditions are evaluated against the configured sensor limits. These scenarios target demo mappings only. They cannot inject readings into a separately registered gateway machine. The selected scenario and run/pause setting survive restart. Changes produce readings immediately when running; paused readings eventually become stale.

A repair report does not switch the scenario to normal. To rehearse post-repair readings, explicitly choose **All within limits**. In a factory, the independent sensor gateway must supply the actual observations.

## Engineer-to-operations workflow

1. Open a machine to inspect its readings, source quality, sensor map and maintenance history. **Ask Copilot** is available throughout the workspace for scoped evidence, approved documentation and investigation assistance.
2. Investigate the issue and create a work-order draft using the approved procedure references. Capture scope, duration, skills, parts, dependencies and due time.
3. A **maintenance manager** reviews the draft. The team proposes a schedule against equipment windows, technician availability, parts and operating restrictions; a manager confirms it.
4. In **Maintenance → Work orders**, open **Field work & team handover**. The approved window and current status are visible. Post coordination notes with **Post team update**; updates are shared only with users authorized to read the work's equipment, sources and procedures.
5. An **engineer or reliability engineer** records the start, with a note and explicit confirmation of coordination with operations, site permits and isolation. Non-simulated work must start within its approved window, and dependent work must already be verified complete. These checks are recordkeeping controls, not a substitute for site authorization.
6. Record the repair and checks performed with **Report issue fixed**. Status becomes **Awaiting verification**. Describe the findings, parts used, post-repair observations and handover; do not mark a task fixed merely because its scheduled time ended.
7. A **different maintenance manager** reads the history and records verification findings. **Verify repair & close work** marks the work **Completed**; **Return for further work** returns it to **In progress**. Administrators cannot bypass this manager role, and a repair reporter cannot self-verify after a role change.
8. The dashboard, asset activity, work queue and schedule show the new status. Completed work leaves upcoming-maintenance summaries but retains its approved schedule and audit history. Copilot can retrieve the current authorized status and recent human notes as cited maintenance history.

In the simulated facility, **Start demo work** permits rehearsal before the planned window. The UI explicitly labels this as a demo; no physical maintenance completion is claimed. Local demo accounts are `engineer / engineer-demo-password` and `manager / manager-demo-password`.

## Collaboration behavior

Work updates append an actor, timestamp, action, notes and monotonic version. Concurrent edits are rejected instead of overwriting a teammate's update. Read the latest history and retry. Maintenance, dashboard and asset views refresh approximately every ten seconds; the submitting view refreshes immediately. Notes in a form remain local until submitted. Verified work is closed; a new issue needs a new work order.

Coordination is recorded within this application. No email, SMS, external CMMS delivery, operations acknowledgement or technician identity-to-login mapping is implied. Existing technician assignments are scheduling resources; authorized engineers record who performed the work in the handover. Production deployment still requires the organization's identity, staffing, commissioning and approval procedures. Sensor condition, risk assessments and alarms remain separate evidence and are not erased by closing work.
