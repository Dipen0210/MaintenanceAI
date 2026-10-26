# SmartVendor inside Inventory

SmartVendor extends PlantPilot's existing counted stock ledger with maintenance-oriented catalog and purchasing workflows. It does not operate equipment, automatically buy goods or supply invented vendor quotations.

## Reviewed parts catalog

Managers and administrators can add a part at zero stock, record manufacturer and manufacturer part number, stock unit, location, reviewed machine compatibility and alternative part IDs. Adding a new planning identifier requires whole-plant asset access and creates a new planning configuration; pending schedules must be proposed again. Existing quantities never become counted receipts.

Units cannot change after stock movements. Reviewed alternatives must belong to the same workspace and use the same unit. Alternatives and compatibility are references for qualified review, not automatic substitution or installation approval. Catalog edits require a reason and current version, and preserve an attributed snapshot.

Minimum and target stock produce replenishment signals using available stock (on hand minus reservations). Accessible approved, unscheduled jobs contribute shortages; no demand or supplier availability is fabricated. Suggestions are planning quantities, not purchase authorization. Lead time is a user-maintained planning value, not a supplier commitment. Suggestions do not include unapproved work or inaccessible work evidence; missing grants can limit the demand shown.

Use **Inventory → Stock movements** for attributed receipts, adjustments and work custody history. Stock administration remains restricted to full-workspace stores roles; engineers record actual parts usage on their assigned jobs.

## Suppliers and quotations

Stores coordinators and managers can record supplier identity/contact references. A maintenance manager approves or suspends each supplier. Supplier reviews are versioned; a new review invalidates older quotations for new purchasing decisions. Create a new supplier record for changed identity/contact details and suspend the old record; approved identity snapshots cannot be silently rewritten.

Record the supplier's quotation reference, stock unit, currency, unit price, fixed freight/tax, valid quantity range, lead time and expiry. A manager checks the source and verifies the draft. Prices are stored as integer minor currency units and displayed in the selected currency. Comparisons include only current verified quotes from currently approved suppliers, within quoted quantities and expiry. Different currencies remain separate; no invented exchange rates, scraped availability or generated supplier ratings are used. Withdraw a superseded quotation and record a fresh one rather than modifying its history.

Quotation verification is a human record that the reference was checked. This increment does not upload or independently authenticate supplier documents, send purchase orders or assert vendor stock availability. The UI preserves retry identifiers for identical submissions so a failed response can be retried without creating another record.
