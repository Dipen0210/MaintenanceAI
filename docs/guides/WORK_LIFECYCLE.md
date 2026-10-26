# Changing unstarted maintenance work

Maintenance managers can cancel an unstarted work order or return a confirmed job to planning to change its window or assigned engineers. Open **Maintenance → Work orders → Schedule changes & cancellation**.

## Before changing work

Confirm with the field team that physical work has not started, enter a reason, and check the confirmation box. These changes are unavailable after any recorded start, including a job subsequently reopened for further repair. If operations has released the equipment, an operations coordinator must first place the handover on **Hold**.

Every change rechecks current role, workspace, machine, source and document permissions. A stale version cannot overwrite another person's start, note or decision. Review the latest record before retrying.

## Reschedule or reassign

1. Choose **Return to planning / reassign**. This immediately releases the old technician, machine-window and planning-parts reservations, removes the old assignment from My Work, and returns the job to Approved.
2. Select the approved schedule candidate, enter the replacement planning window, and optionally select eligible technicians linked to enabled engineer accounts. Leaving the selection empty uses the configured planning resources.
3. Generate a feasible proposal. Skills, availability, parts, dependencies and machine windows still apply. The chosen team is not guaranteed a feasible result.
4. Obtain any required operations approval for the exact replacement window, then confirm the new schedule as a maintenance manager. Operations must record a fresh release before the engineer starts when coordination is required.

The old reservation is not retained while a replacement is prepared. Historical proposals retain their original decision but label released assignments as inactive. They cannot authorize execution. Prior windows, resources, assigned users, reasons and operations decisions remain in the work order's schedule change history.

## Cancel

Choose **Cancel unstarted work** for a draft, approved or scheduled job that has never started. Cancellation closes that work record and releases its unused reservations. It does not resolve a linked issue, clear sensor alerts, record consumed inventory or operate equipment. There is no undo action; create and review separate work if needed.

Open dependent work blocks cancellation. Confirmed or active dependent work also blocks returning a prerequisite to planning. Pending proposals containing the changed job are rejected as superseded; generate a fresh proposal. Cancelled calendar or meter occurrences remain recorded, so a worker retry cannot recreate the same occurrence; later due occurrences remain eligible.

## Boundaries

This workflow does not transfer active repairs, provide an atomic swap that retains the old reservation until a replacement is approved, send external notifications, or alter physical equipment. Workspaces with [counted inventory](INVENTORY.md) require issued parts to be returned before unstarted cancellation/replanning and release recorded stock reservations. Other workspaces retain planning quantities. Local tests and synthetic rehearsals do not establish factory commissioning or qualified Copilot approval.
