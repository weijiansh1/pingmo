# A116 Formal G0 Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, per-aircraft GJB Figure A116 audit for all 3,000 IV-A raw plants using the prescribed 60-degree-in-`1.7*T_d` held-step condition and strict peak-valley-peak extraction.

**Architecture:** Keep waveform/peak logic in `src/benchmark/time_domain.py`, where it can be unit-tested independent of A116 boundaries. `src/quality/a116.py` will own per-aircraft step-amplitude calibration and status-bearing A116 assessment. The existing audit script will aggregate the strictly assessed records and plot only valid comparison points.

**Tech Stack:** Python 3.14, NumPy, SciPy, pytest, Matplotlib, JSON Lines.

---

## File structure

- Modify `src/benchmark/time_domain.py`: add signed roll-rate extrema extraction, A120 ratio validation, and a full plant held-step simulator for amplitude calibration.
- Modify `src/quality/a116.py`: calibrate a per-plant held step, run the no-spiral response at that amplitude, and return a status-bearing formal audit record.
- Modify `tests/test_metrics.py`: specify peak-valley-peak and invalid-extraction behavior.
- Modify `tests/test_a116.py`: specify a reachable formal audit and a deliberately unreachable audit.
- Modify `scripts/06_audit_a116_iv_a.py`: aggregate reasons, label formal results, and show valid points versus A116 boundaries.
- Modify `docs/manual_alignment_status.md`: replace the obsolete P1/P2/P3 missing-definition statement with the remaining G0 result state.

### Task 1: Strict A120 waveform primitives

**Files:**
- Modify: `src/benchmark/time_domain.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing peak-valley-peak test**

```python
from src.benchmark.time_domain import extract_gjb_roll_peaks


def test_extract_gjb_roll_peaks_uses_first_peak_first_valley_second_peak() -> None:
    time = np.arange(7.0)
    roll_rate = np.array([0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0])

    peaks = extract_gjb_roll_peaks(time, roll_rate)

    assert peaks.status == "assessable"
    assert (peaks.p1, peaks.p2, peaks.p3) == pytest.approx((2.0, 1.0, 3.0))
    assert (peaks.t1_s, peaks.t2_s, peaks.t3_s) == pytest.approx((1.0, 2.0, 3.0))
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `pytest tests/test_metrics.py::test_extract_gjb_roll_peaks_uses_first_peak_first_valley_second_peak -v`

Expected: FAIL because `extract_gjb_roll_peaks` does not exist.

- [ ] **Step 3: Add the minimal peak record and extraction implementation**

```python
@dataclass(frozen=True, slots=True)
class GJBRollPeaks:
    status: str
    reason: str | None
    p1: float | None
    p2: float | None
    p3: float | None
    t1_s: float | None
    t2_s: float | None
    t3_s: float | None


def extract_gjb_roll_peaks(time: np.ndarray, roll_rate: np.ndarray) -> GJBRollPeaks:
    maxima, _ = find_peaks(roll_rate)
    if len(maxima) < 2:
        return GJBRollPeaks("not_assessable", "missing_two_roll_rate_peaks", None, None, None, None, None, None)
    first_peak, second_peak = int(maxima[0]), int(maxima[1])
    minima, _ = find_peaks(-roll_rate[first_peak:second_peak + 1])
    if not len(minima):
        return GJBRollPeaks("not_assessable", "missing_valley_between_peaks", None, None, None, None, None, None)
    valley = first_peak + int(minima[0])
    return GJBRollPeaks("assessable", None, float(roll_rate[first_peak]), float(roll_rate[valley]), float(roll_rate[second_peak]), float(time[first_peak]), float(time[valley]), float(time[second_peak]))
```

Validate the A120 denominator in `gjb_roll_oscillation_ratio`: raise `ValueError` when it is non-positive instead of replacing it with epsilon.

- [ ] **Step 4: Run the targeted test and verify GREEN**

Run: `pytest tests/test_metrics.py::test_extract_gjb_roll_peaks_uses_first_peak_first_valley_second_peak -v`

Expected: PASS.

- [ ] **Step 5: Write and run the missing-valley RED/GREEN regression**

```python
def test_extract_gjb_roll_peaks_reports_no_second_peak_without_inventing_values() -> None:
    peaks = extract_gjb_roll_peaks(np.arange(4.0), np.array([0.0, 1.0, 0.7, 0.5]))
    assert peaks.status == "not_assessable"
    assert peaks.reason == "missing_two_roll_rate_peaks"
    assert peaks.p1 is None
```

Run: `pytest tests/test_metrics.py -v`

Expected: all metric tests PASS.

### Task 2: Per-aircraft standard-step calibration

**Files:**
- Modify: `src/benchmark/time_domain.py`
- Modify: `src/quality/a116.py`
- Test: `tests/test_a116.py`

- [ ] **Step 1: Write the failing calibration tests**

```python
from src.quality.a116 import calibrate_a116_step


def test_calibrate_a116_step_hits_60_degrees_at_1_point_7_dutch_roll_time_constants() -> None:
    parameters = PChannelParameters(1.0, -0.04, 0.5, 0.2, 2.0, 1.5, 0.7, 0.0125)
    calibration = calibrate_a116_step(parameters, max_step_force_n=22.0, dt=0.005)

    assert calibration.status == "assessable"
    assert calibration.target_time_s == pytest.approx(1.7 / (parameters.zeta_d * parameters.omega_d))
    assert calibration.bank_angle_at_target_deg == pytest.approx(60.0, abs=0.05)
    assert 0.0 < calibration.step_force_n <= 22.0
```

- [ ] **Step 2: Run the calibration test and verify RED**

Run: `pytest tests/test_a116.py::test_calibrate_a116_step_hits_60_degrees_at_1_point_7_dutch_roll_time_constants -v`

Expected: FAIL because `calibrate_a116_step` does not exist.

- [ ] **Step 3: Implement a linear, delay-aware held-step calibration**

Add `held_step_roll_rate_response(parameters, time, force_n)` to `time_domain.py`. It must instantiate `PChannel`, apply `force_n` at every timestep, and return the sampled roll rate; use cumulative trapezoidal integration to derive bank angle.

In `a116.py`, add `A116StepCalibration` and `calibrate_a116_step`. Use a unit-force simulation at a horizon that includes `1.7 / (zeta_d * omega_d)`. Interpolate its bank angle at that exact target time and set
`force_n = 60 / phi_unit_deg`. Return `input_not_achievable` when the unit response is non-positive or the force exceeds `max_step_force_n`; otherwise perform a scaled response and record its target error.

- [ ] **Step 4: Run the calibration test and verify GREEN**

Run: `pytest tests/test_a116.py::test_calibrate_a116_step_hits_60_degrees_at_1_point_7_dutch_roll_time_constants -v`

Expected: PASS.

- [ ] **Step 5: Add and verify the non-achievable-input regression**

```python
def test_calibrate_a116_step_does_not_assign_force_when_60_degree_target_exceeds_limit() -> None:
    parameters = PChannelParameters(0.001, -0.04, 0.5, 0.2, 2.0, 1.5, 0.7, 0.0125)
    calibration = calibrate_a116_step(parameters, max_step_force_n=22.0, dt=0.005)

    assert calibration.status == "not_assessable"
    assert calibration.reason == "input_not_achievable"
    assert calibration.step_force_n is None
```

Run: `pytest tests/test_a116.py -v`

Expected: all A116 tests PASS.

### Task 3: Formal A116 audit record and classification

**Files:**
- Modify: `src/quality/a116.py`
- Modify: `tests/test_a116.py`

- [ ] **Step 1: Replace the old unvalidated-audit expectation with a failing formal-audit test**

```python
def test_a116_audit_reports_strict_peaks_and_never_uses_the_legacy_proxy() -> None:
    audit = audit_a116_parameters(boundaries, parameters, "A_C", np.arange(4000) * 0.005)

    assert audit["a116_status"] in {"assessable", "not_assessable"}
    assert "p_osc_over_p_av_unvalidated_peak_proxy" not in audit
    if audit["a116_status"] == "assessable":
        assert audit["p1"] is not None
        assert audit["p2"] is not None
        assert audit["p3"] is not None
        assert audit["a116_level"] in {1, 2, "above_level_2", "not_available"}
```

- [ ] **Step 2: Run the formal-audit test and verify RED**

Run: `pytest tests/test_a116.py::test_a116_audit_reports_strict_peaks_and_never_uses_the_legacy_proxy -v`

Expected: FAIL because the current audit exposes the unvalidated proxy and fixed `not_available` status.

- [ ] **Step 3: Implement status-first formal audit behavior**

Make `audit_a116_parameters` call `calibrate_a116_step`. If calibration is not assessable, return `a116_status="not_assessable"`, its reason, calibration fields, and `a116_level="not_available"`.

For an assessable calibration, multiply the no-spiral unit response by the calibrated force, call `extract_gjb_roll_peaks`, calculate the validated ratio, then call `assess_a116`. Preserve source pages and `spiral_mode_removed=True`. Do not emit `p_osc_over_p_av_unvalidated_peak_proxy`.

- [ ] **Step 4: Run the targeted audit test and verify GREEN**

Run: `pytest tests/test_a116.py::test_a116_audit_reports_strict_peaks_and_never_uses_the_legacy_proxy -v`

Expected: PASS.

- [ ] **Step 5: Run both focused test modules**

Run: `pytest tests/test_metrics.py tests/test_a116.py -v`

Expected: PASS with no legacy-proxy failures.

### Task 4: G0 reporting, smoke run, and full audit

**Files:**
- Modify: `scripts/06_audit_a116_iv_a.py`
- Modify: `docs/manual_alignment_status.md`

- [ ] **Step 1: Update the report schema and plot contract**

Change report metadata to state the held-step condition, `max_step_force_n=22.0`, and strict P1/P2/P3 method. Count `a116_status`, `a116_level`, and `a116_reason`. In the lower plot, scatter only rows with `a116_status == "assessable"`; label all other rows by reason in a bar chart or annotation. Remove every "未验证峰值代理" label.

- [ ] **Step 2: Run a 16-plant smoke audit**

Run: `python scripts/06_audit_a116_iv_a.py --limit 16 --duration 20 --output results/tmp_a116_smoke.json`

Expected: exit 0; report has 16 plants; `img/手册G0_A116_IV-A原始飞机审计_烟测.png` exists.

- [ ] **Step 3: Visually inspect the smoke plot and verify schema**

Run: `python -c "import json; r=json.load(open('results/tmp_a116_smoke.json', encoding='utf-8')); assert r['plant_count']==16; assert all('a116_status' in p for p in r['plants']); print(r['level_counts'])"`

Inspect: `img/手册G0_A116_IV-A原始飞机审计_烟测.png`.

Expected: no points are shown as levelled without an `assessable` status; failure causes are visible.

- [ ] **Step 4: Run the full 3,000-plant audit**

Run: `python scripts/06_audit_a116_iv_a.py --duration 20 --output results/手册G0_A116_IV-A审计.json`

Expected: exit 0; report plant count is 3000; plot is `img/手册G0_A116_IV-A原始飞机审计.png`.

- [ ] **Step 5: Verify output integrity and update gate documentation**

Run:

```powershell
python -c "import json; r=json.load(open('results/手册G0_A116_IV-A审计.json', encoding='utf-8')); assert r['plant_count']==3000; assert len(r['plants'])==3000; assert all('a116_status' in p for p in r['plants']); print(r['level_counts'])"
pytest -q
```

Update `docs/manual_alignment_status.md` to say that P1/P2/P3 now use the first-peak, first-valley, second-peak definition, and record the resulting G0 status exactly as produced. Do not mark G0 passed unless the output, source coverage, and manual gate criteria demonstrate it.

- [ ] **Step 6: Commit (only if a Git repository exists)**

Run: `git rev-parse --is-inside-work-tree`

If it reports `true`, commit only the A116 implementation, tests, report script, documentation, and intentional G0 artifacts. If it reports an error, record that this workspace is not a Git repository and leave files uncommitted.

## Plan self-review

- Spec coverage: Tasks 1-3 cover strict peak/valley/peak extraction, the 60-degree held-step condition, no-spiral A120 metric, source-boundary classification, and all specified non-assessable cases. Task 4 covers the required smoke and full 3,000-aircraft outputs.
- Placeholder scan: no unfinished markers or undefined follow-up task remains.
- Type consistency: `GJBRollPeaks`, `A116StepCalibration`, `calibrate_a116_step`, and `a116_status` are introduced before consumers use them; formal records retain existing `a116_level` compatibility.
