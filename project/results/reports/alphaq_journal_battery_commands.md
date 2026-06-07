# AlphaQ Journal Battery Commands

Battery CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_journal_next_battery.csv`.

These commands are generated from the journal evidence audit. Run them from `/Users/caio/Deep-Learning` when the Apuana environment is reachable. They intentionally keep outputs separated by suffix so the current baseline artifacts are not overwritten.

## tensor-v3-screen

Targets: `gf_2pow5_mult,nc_tof_5,cuccaro_adder_n5`.

Blocked: Run a tensor-v3/profile screening step first; do not repeat full-action objective-grid jobs for this target without a new screening signal.

## restricted-action-pilot

Targets: `hwb_6`.

Blocked: No restricted-action submit path is implemented yet.
