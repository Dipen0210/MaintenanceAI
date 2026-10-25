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

## Workspace views and field issue intake

Use the **Workspace** selector in the sidebar to choose **Demo facility** or **Plant workspace**. The demo view contains equipment with simulation mappings; the plant view contains other registered equipment. Both views retain the signed-in user's existing site and resource grants. This selector is a data-origin filter, not new company provisioning or proof of production commissioning. Administration and shared planning resources remain explicitly site-wide. Knowledge/model evaluation tools may contain site-wide reference information; asset evidence remains authorized separately.

Switching views clears the selected machine, page state and Copilot context. Requests and telemetry streams carry the selected view; a delayed response from the previous view is discarded. The choice persists for that local user. Pausing the simulation leaves recorded demo history in the demo view. Select Plant workspace to inspect registered equipment separately. A registered machine with no fresh measurements remains unverified. Simulation-only procedures cannot apply to physical equipment, and demo planning resources cannot schedule physical maintenance.

An operator, engineer or other authorized user can open **Issues → Report an issue** (or **Report issue** on a machine). Select the machine, describe the observed symptom and record reported urgency. Reporting needs asset access, but does not require telemetry, model output or a risk assessment. Urgency is a human report, not a measured condition. Reports do not create approved work, clear sensor warnings or notify an external recipient.

A maintenance manager opens the report, records triage notes and assigns an enabled engineer with access to that machine. The engineer sees it in **My Work → Assigned investigations**, starts the investigation and records notes. **Ask Copilot about machine** selects the report's machine; issue text/photos are not automatically included in Copilot's diagnostic evidence. Physical maintenance continues through the separately reviewed work-order flow. Only a manager may close or reopen the investigation; closing it does not mean the machine has been repaired.

The reporter, assigned engineer or manager may attach up to five JPEG/PNG photos, each at most 5 MiB and 12 megapixels. Photos are normalized, resized to a maximum 2,000-pixel dimension, stripped of embedded metadata, and downloaded through authenticated asset access. They are not public URLs. Updates retain actor, timestamp, notes and version. Conflicting changes require a refresh instead of overwriting another user's work.

## Named maintenance assignments

In **Maintenance → Resources**, a manager or administrator may bind each technician resource to an enabled engineer account. One account can represent at most one resource in a configuration. At schedule confirmation, the backend rechecks the account's role and all evidence access and freezes the selected user assignments with the approved schedule. If any assignment is invalid, no part of the schedule is reserved.

Assigned jobs appear in **My Work → Approved maintenance jobs**. For work with named assignments, only an assigned engineer may record start or report a repair; independent manager verification remains required. Revoked access removes affected work from the user's queue and blocks changes. Existing unmapped demonstration resources retain the shared engineering workflow and do not populate personal queues. Updating a resource configuration does not reassign an already confirmed job.

This increment does not yet implement QR scanning, recurring preventive work, operations-specific acknowledgements, reassignment of confirmed schedules, external notifications, offline synchronization, inventory transactions or production tenant provisioning. Expert review remains deferred; local software tests are not factory diagnostic validation.

## Machine QR access

Open an authorized machine and expand **Machine QR label** to download its PNG. Print the machine ID beside the code. A phone camera opens the machine link after sign-in; the workspace resolves the correct Demo/Plant view without changing grants. **Scan machine** also offers browser camera detection when supported and manual ID/link entry when unavailable or denied. Codes contain no credentials and do not authorize access.

Localhost labels only work on the computer running the app. A reachable deployment can set `MAINTENANCEAI_PUBLIC_WEB_ORIGIN` to its approved origin (scheme, host and optional port, no path). This allowlists QR generation only; configure the deployment's HTTPS, authentication, proxy/CORS/origin policy and network access separately. No remote deployment or physical camera compatibility is established by the local tests.

## Recurring preventive work

A maintenance manager opens **Preventive plans → New preventive plan**, selects equipment and approved procedures, then supplies the task, calendar, timezone, resource requirements and change reason. Calendar intervals support days, weeks and months. The backend checks due plans every minute while running, including a startup pass. **Generate due work** also runs a bounded pass immediately.

Each due occurrence creates one draft work order. It needs the existing manager review and confirmed schedule before field execution; no resources are automatically reserved. The calendar basis is explicit and does not invent a risk assessment. Use the generated job link to review, schedule and execute it. Independent verification updates the plan's work history without clearing sensor alarms.

Monthly recurrence uses the original day or the last day in a shorter month. Daylight-saving gaps use the first valid minute, repeated times the first occurrence. Pause/resume requires a reason and retains missed dates; catch-up creates at most 20 drafts per plan per pass. Repeated runs cannot duplicate a due date. Revisions preserve generated jobs. A new calendar anchor cannot be more than 31 days in the past. Missed occurrences are not automatically discarded.

The plan shows next generation, blocked generation, revision history and the latest 100 jobs. Jobs become overdue after the configured grace period until completed. Disabled/revoked manager access or unavailable approved procedures blocks generation; a currently authorized manager must review and revise the plan. Meter-based maintenance, inspection rounds, holiday/shutdown scheduling and notifications are not included in this slice.
