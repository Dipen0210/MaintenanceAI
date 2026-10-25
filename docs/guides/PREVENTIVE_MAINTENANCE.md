# Preventive maintenance and field inspections

Open **Preventive plans** in the Operations workspace. Its views are **Calendar plans**, **Meter plans** and **Inspection rounds**. All views retain the current Demo or Plant context and backend equipment/document permissions.

## Calendar plans

A maintenance manager configures an approved task, machine, first local due date, IANA timezone, daily/weekly/monthly interval and grace period. The backend checks due work at startup and every minute; **Generate due work** runs an immediate check. Each occurrence creates an unapproved work order. Existing approval, resource scheduling, execution and repair-verification steps still apply.

Monthly schedules preserve their original day, clamping shorter months. A daylight-saving gap uses the next valid minute; repeated times use the first occurrence. Pausing retains missed dates; resuming catches up in batches of 20. Previously generated dates are deduplicated. Changing a calendar does not rewrite existing work.

## Meter plans

1. An administrator opens **Meter plans → Configure meter**, selects an existing machine and data source, and maps its cumulative operating-hours (`h`) or cycles tag. Configure its valid raw range and freshness interval. Mapping does not install a sensor or establish a connection. Demo and Plant origins cannot be mixed.
2. Supply observations through the configured read-only gateway or file-import workflow. Use the exact mapped tag, unit and source. No cumulative reading is fabricated by creating a mapping or plan.
3. A maintenance manager creates a meter plan with its first threshold, repeat interval, approved references, task, staffing, parts and grace period.
4. The minute worker or **Check meter now** examines fresh good-quality observations. Reached thresholds create at most 20 draft jobs per pass. Detection time is recorded as due time; actual threshold-crossing time is unknown. A stopped counter produces no additional threshold occurrence.
5. Review and schedule the resulting drafts under **Maintenance**. Engineers execute approved work and a manager independently verifies the reported repair.

Meter identity, first threshold and interval are immutable after creation. Revise task details with an attributed reason, or pause the old plan and create a new one for a new recurrence. Consider existing generated work when configuring catch-up thresholds. Old job snapshots remain unchanged.

A decreasing counter blocks generation. A manager must review the exact blocked sample and record either a **reset/replacement** with an explicitly confirmed cumulative total, or a **rollover** with the counter's verified modulus. Reset totals cannot reduce the previous accepted total. Version checks prevent conflicting reviews. The offset and reconciliation history remain visible; neither establishes physical condition. Stale, missing or invalid readings, disabled owners and revoked references/access block generation.

## Inspection rounds

A maintenance manager creates a versioned ordered checklist with up to 20 checks. Each check names a machine and approved references, and uses either pass/fail or a numeric measurement with units and accepted limits. Every saved result needs evidence notes. A skipped check needs a reason and cannot claim a measurement.

The manager assigns a template version and due date to an enabled engineer with access to every machine and reference. It appears in **My Work**. An assignment does not approve corrective work or downtime. The engineer confirms site clearance for the inspection scope, starts the round, then records or explicitly skips each check in order. Saved results persist across reloads. Template revisions do not alter already assigned rounds.

An out-of-range measurement or failed check creates one linked issue for triage. Repeated saves do not duplicate it. Correcting a reading preserves its earlier audit record and leaves the issue open for explicit follow-up. Add photos through the existing linked-issue workflow. Observations and their source events can be cited by Copilot, with current access checks and historical context. **Ask Copilot about this machine** selects that check's machine context.

Completion requires all checks to be recorded or skipped. A round with skipped checks is visibly **completed with skips**. Completed and cancelled records are immutable; assign a new round for follow-up. A manager can cancel an open round with a reason. Inspection completion does not clear sensor alerts, confirm repairs or establish machine health.

## Current boundaries

This is a local, online workflow with bounded generation/history lists. It does not yet include offline synchronization, recurring assignment of inspection rounds, required per-check photo uploads, automatic holiday/shutdown policies, external operations handovers or notification escalation. Gateway delivery/commissioning and industrial diagnostic accuracy need separate validation. Qualified Copilot review remains deferred; software tests and synthetic examples are not factory acceptance evidence.
