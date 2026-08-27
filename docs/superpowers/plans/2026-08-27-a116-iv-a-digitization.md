# A116 IV-A Digitization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the IV-A roll-rate-oscillation gate traceable to GJB 2874-97 Figure A116 without fabricating a scalar threshold.

**Architecture:** Store the A/C Level 1 and Level 2 boundary points manually digitized from PDF page 240 (printed page 236) in a source-labelled CSV. Compute the permitted `psi_p` proxy from the complex Dutch-roll residue of the existing P-channel step response, then interpolate the selected boundary and report a level only within its source domain.

**Tech Stack:** Python, NumPy, SciPy, pytest, Matplotlib.

---

### Task 1: Source-labelled A/C curve data

**Files:** Create `data/gjb_a116_boundary.csv`; test `tests/test_a116.py`.

- [x] Write a failing test that reads the `A_C` Level 1/2 curves and checks their source coordinates and interpolation at a knot.
- [x] Add digitized A/C boundary points with columns `phase_group,level,psi_p_deg,p_osc_over_p_av,source_pdf_page,source_print_page,method` and an explicit `manual_trace_from_scanned_figure` method.
- [x] Run the test and verify it passes.

### Task 2: A116 evaluator and roll-rate phase proxy

**Files:** Create `src/quality/a116.py`; test `tests/test_a116.py`.

- [x] Write a failing test for level classification below/above the A/C boundaries and for a wrapped `[-360, 0)` `psi_p` result.
- [x] Implement CSV loading, linear interpolation only over the traced domain, and the Dutch-roll complex-pole residue calculation of `psi_p` for a unit step. Return `not_available` outside the curve domain.
- [x] Run the focused test and verify it passes.

### Task 3: IV-A audit output

**Files:** Create `scripts/06_audit_a116_iv_a.py`; modify `docs/manual_alignment_status.md`; test `tests/test_a116.py`.

- [x] Write a failing test that evaluates a real P-channel parameter set and produces an A116 report with raw metric, `psi_p`, source and status.
- [x] Add the script to read the IV-A plant bank, calculate the phase proxy and raw peak metric, and write a JSON/PNG audit with source labels. It must not label a plant outside the digitized domain.
- [x] Run the focused test, then the full suite, and inspect the generated plot before changing Gate status.
