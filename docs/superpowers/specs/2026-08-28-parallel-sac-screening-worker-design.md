# Parallel SAC Screening Worker Design

## Goal

Run selected independent SAC screening runs concurrently on the same lightly loaded P4, without changing their experimental definitions or duplicating work already owned by the sequential batch runner.

## Scope

Add a command-line worker that receives exactly one existing batch `run_id`. It reconstructs that run from the frozen configuration, trains it, evaluates the same six held-out aircraft, and writes the existing `screening_report.json` artifact format.

The first use will launch only the two disjoint `multi-strong-40k` seed runs while the current sequential process continues. These are deliberately last in the sequential ordering: by the time that process reaches them, completed reports will cause its existing resume logic to skip them.

## Interface and data flow

`scripts/09_run_gpu_sac_screening_worker.py --run-id <id>` will:

1. Map the ID to one of the four frozen configurations and two frozen seeds in `08_run_gpu_sac_screening_batch.py`.
2. Refuse an unknown ID and exit non-zero.
3. Exit successfully without training when a valid completed report is already present.
4. For an incomplete valid run, use the same library, correction ratio, model construction, training function, held-out evaluation, metric summary, and JSON schema as the batch runner.

The worker writes only its own `checkpoints/gpu_sac_screening_batch/<run-id>/` directory. It does not write the aggregate summary, touch another run directory, or alter the ongoing sequential runner.

## Safety and reproducibility

- Every run retains its frozen seed, plant set, reward weights, 40k/120k budget, and CUDA device choice.
- Launchers use separate log files and only the two final strong-penalty run IDs.
- A completed report is the resume marker; no report is ever overwritten.
- The aggregate sequential runner remains the sole producer of `results/GPU批量SAC筛选报告.json` after it has collected or skipped all eight runs.

## Verification

Unit tests will prove that valid IDs resolve to the frozen configuration and seed, an unknown ID is rejected, and an already-completed report is skipped. Before remote launch, run the relevant test file and the full local suite. On the P4, verify both worker PIDs, distinct run directories, separate logs, GPU process count, and eventual reports.
