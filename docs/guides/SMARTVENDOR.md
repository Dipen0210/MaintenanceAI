# SmartVendor inside Inventory

SmartVendor extends PlantPilot's existing counted stock ledger with maintenance-oriented catalog and purchasing workflows. It does not operate equipment, automatically buy goods or supply invented vendor quotations.

## Reviewed parts catalog

Managers and administrators can add a part at zero stock, record manufacturer and manufacturer part number, stock unit, location, reviewed machine compatibility and alternative part IDs. Adding a new planning identifier requires whole-plant asset access and creates a new planning configuration; pending schedules must be proposed again. Existing quantities never become counted receipts.

Units cannot change after stock movements. Reviewed alternatives must belong to the same workspace and use the same unit. Alternatives and compatibility are references for qualified review, not automatic substitution or installation approval. Catalog edits require a reason and current version, and preserve an attributed snapshot.

Minimum and target stock produce replenishment signals using available stock (on hand minus reservations). Accessible approved, unscheduled jobs contribute shortages; no demand or supplier availability is fabricated. Suggestions are planning quantities, not purchase authorization. Lead time is a user-maintained planning value, not a supplier commitment. Suggestions do not include unapproved work or inaccessible work evidence; missing grants can limit the demand shown.

Use **Inventory → Stock movements** for attributed receipts, adjustments and work custody history. Stock administration remains restricted to full-workspace stores roles; engineers record actual parts usage on their assigned jobs.
