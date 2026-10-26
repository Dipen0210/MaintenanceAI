# Counted spare-parts inventory

Open **Inventory** as a maintenance manager, administrator or stores coordinator with access to every machine in the selected workspace. Assigned engineers record usage inside their maintenance jobs. Demo and Plant have separate stock balances.

## Enable tracking

A manager or administrator first enables counted inventory and records a stocktake setup reason. Activation starts every configured part at **zero**; planning estimates are never imported as counted stock. Record opening quantities through **Receive stock** after counting them. Tracking remains enabled for that workspace.

Finish existing parts-bearing scheduled or active work before activation, or return unstarted jobs to planning. This prevents existing jobs from acquiring invented opening reservations or consumption. Workspaces that have not activated inventory continue using planning quantities; those estimates do not constitute a physical stock ledger.

Add part identifiers through **Maintenance → Planning resources**. New catalog entries start at zero. Editing a planning quantity cannot change counted stock. Use a receipt or a reasoned count adjustment instead. Quantities are whole units; retain a consistent physical unit for each identifier.

## From stores to verified work

1. Receive counted stock. On hand records the quantity in stores; available equals on hand minus reserved.
2. Review the work scope and confirm a feasible schedule. Confirmation checks current stock and reserves the approved quantity in the same transaction. A proposal alone does not reserve stock.
3. In the work order, open **Parts & reconciliation → Record a parts movement**. Authorized stores staff issue reserved parts to the job. Issuing reduces both on hand and reserved, and records unsettled custody at the job.
4. Once the job starts, its named engineer records the quantity actually used. Managers and stores coordinators may also record usage. This reduces unsettled custody and increases recorded consumption; it does not remove stock from stores a second time.
5. Stores staff return unused issued parts. Returned quantities become available in stores, without automatically becoming reserved again.
6. An independent maintenance manager verifies the repair. All issued parts must first be consumed or returned. Verification releases any unused reservation. Operations approval for return to production remains a separate decision where required.

Cancellation or replanning of unstarted work requires issued parts to be returned first and releases unused reservations. Reopening a verified repair awaiting operations return reserves the remaining approved allowance again; insufficient stock blocks the entire reopen, including its handover changes. No movement expands the approved parts allowance. Further material requirements need separately reviewed work scope.

## Records and permissions

Every movement records its actor, reason, quantity and timestamp. Work-order histories show job-specific custody and usage; the shared inventory page shows balances and the latest 200 movements, with work entries filtered by current evidence access. Corrections use new movements; the API does not edit or delete earlier movements.

Receipts and adjustments use the current stock version. Job movements use the same version as field updates, preventing an older repair-verification form from overlooking a new issue or return. Identical command retries return the original movement; changed details cannot reuse its request key. Current access is checked again before accepting a retry. Concurrent transactions cannot over-reserve stock or make balances negative.

Stores coordinators cannot approve schedules, verify repairs or authorize equipment operation. Inventory administration requires full machine access within its workspace; work transactions additionally require current machine, source and procedure access. Engineers can consume only on their named assignments.

## Current limits

This is a minimal ledger, not a warehouse or purchasing system. It does not provide bins, lots, serial tracking, fractional quantities, costing, reorder automation, purchase orders, ERP synchronization, cross-site transfers or stocktake reconciliation against a physical system. Incorrect consumption requires an attributed correction process outside this first increment; it cannot be erased through the UI. No stock movement operates equipment or proves a real factory stocktake occurred. Local automated tests and synthetic browser rehearsals do not establish industrial commissioning.
