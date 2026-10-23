# Maintenance planning and approval

The application stores work orders and approved schedules internally in SQLite. It never writes to an external CMMS, SCADA server, PLC or actuator. The copilot's work-order and schedule tools remain read-only.

`006_scheduling.sql` introduces versioned plant resource configurations, immutable work-order content, proposal snapshots, review decisions and confirmed resource reservations. A draft links a current authorized risk assessment, applicable approved inspection passages and optionally the engineer's saved investigation. The backend assigns the assessment's priority; clients cannot override it. The risk snapshot remains explicitly historical during later planning. It is not presented as a fresh measurement or a failure probability.

Only a maintenance manager can approve or reject a candidate, then confirm or reject a proposed schedule. Administrator privileges alone do not approve maintenance. A confirmed schedule reserves resources; it does not assert that a technician performed the work or that equipment is safe to operate. Site isolation, permits and qualified review remain necessary.

## Constraint model

The actual implementation uses OR-Tools CP-SAT interval variables, optional technician/team assignments, precedence and non-overlap constraints. Primary references: [job-shop scheduling](https://developers.google.com/optimization/scheduling/job_shop) and [CP-SAT solver statuses](https://developers.google.com/optimization/cp/cp_solver).

All times are normalized UTC and solved at one-minute resolution. Availability starts round up and ends round down; occupied intervals round outwards. The horizon is at most 14 days, with at most 20 selected work orders, 12 technicians, 100 windows and 2,000 candidate team/window assignments per order. Teams contain one to three technicians; their combined skills must cover the work's requirements. Every technician must be available for the full task.

Hard constraints cover skills, full working periods, equipment windows, operational blackouts, task duration, parts quantity and delivery time, dependencies, technician overlap and equipment overlap. Previously confirmed plant-wide reservations are considered even when the requesting engineer cannot view those other assets. Private reservation details are not exposed in the proposal. Overlapping fixed blackouts/reservations are merged before modeling.

The objective minimizes weighted completion time, five times weighted overdue minutes, configured disruption units and makespan. Priority weights are critical 1,000, high 100, insufficient evidence 100, planned 10 and monitor 1. Insufficient evidence is an explicit verification priority, not an inferred fault probability. These are transparent demonstration preferences, not calibrated financial or safety costs. Disruption penalties can outweigh delay preferences; managers must inspect assignments and due times. No objective term can relax a hard constraint.

One search worker and seed 42 make fixed-input search deterministic within a two-unit deterministic search budget; a 15-second wall-clock guard also applies. An optimal or feasible result must pass a separate feasibility validator. An actual proof of infeasibility returns blocking constraints; a search/candidate limit returns `unknown`, never a false infeasibility claim. Joint infeasibility identifies the interacting constraint categories; the current solver does not extract a minimum unsatisfiable core.

## Transactional approval and access

Drafts and proposals do not reserve resources. Final approval uses `BEGIN IMMEDIATE`, rechecks the current enabled manager role, every asset/source/procedure grant, applicable document approval and linked investigation evidence. It rejects changed resource configuration, changed/already-scheduled candidates, elapsed start times and newly conflicting reservations. It independently validates the exact proposed assignments against current reservations inside that transaction, then writes all assignments and work-order status changes atomically. Concurrent or duplicate approvals cannot double-book resources.

Resource configuration requires manager/administrator authority and grants to all assets in the plant, preventing a partially scoped user from removing other assets' restrictions. Reads filter equipment windows and restrictions to authorized assets. Technician and parts configuration is shared plant planning information. Revoked evidence withholds affected work orders/proposals and prevents their approval. Audit events retain actor, action, resource, timestamp and decision reason.

Stock quantities represent the total planning allocation, and confirmed assignments permanently reserve their quantities. This version does not claim inventory replenishment reconciliation, maintenance completion, schedule cancellation or ERP capabilities. Changing resources creates a new version and invalidates pending proposals; it does not silently rewrite previously approved reservations. Users must preserve existing commitments when reconciling resource configuration.

## Application flow

Operations → actual synthetic incident inference → risk assessment → cited copilot investigation → saved investigation → maintenance draft → manager candidate review → feasible proposal → manager confirmation. `/operations/scheduling` also supports direct evidence-linked drafts, resource/parts/windows/restrictions editing, dependencies, explicit conflicts and persisted confirmed schedules. Every proposed task requires approved inspection references; free text is a proposed scope, not a new approved procedure.

Demo initialization creates clearly simulated technicians, ten inspection kits and next-day 08:00–16:00 UTC windows, with an 11:00–12:00 conveyor production restriction. It creates no approved work. When those windows become historical, a manager must configure new periods; restarting does not silently change the planning calendar.

Run `venv/bin/python -m scripts.evaluate_scheduling` to reproduce six authored synthetic scenarios. Both greedy baselines enforce the same hard constraints, but do not backtrack earlier choices. Their failure is explicitly not proof of infeasibility. The report measures feasibility, constraint violations, overdue priority work, configured disruption and utilization. It does not measure industrial downtime, savings or reliability improvement.
