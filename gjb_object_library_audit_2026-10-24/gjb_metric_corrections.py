"""Scoped GJB 2874-97 roll metrics; no invented peaks or full-quality rating.

Source: supplied PDF page 12 / printed page 8, section 3.6.
Caller must provide a response with the spiral contribution removed, the proper
input sign, and sufficient sampling. A116 grading additionally needs beta phase.
"""
import math

import numpy as np
from scipy.signal import find_peaks


def roll_oscillation_ratio(time, response, zeta_d, *, spiral_removed, command_sign=1, prominence=0.0):
    t, p = np.asarray(time, float), np.asarray(response, float)
    if t.ndim != 1 or p.shape != t.shape or len(t) < 3 or not np.isfinite(t).all() or not np.isfinite(p).all() or not np.all(np.diff(t) > 0):
        raise ValueError("Require finite, increasing time and a same-length finite response")
    if not math.isfinite(zeta_d) or not 0 <= zeta_d < 1 or command_sign not in (-1, 1) or prominence < 0:
        raise ValueError("Invalid damping, command sign or prominence")
    if not spiral_removed:
        return {"status":"not_assessable", "reason":"spiral_component_not_removed", "ratio":None}
    p = p*command_sign
    peaks, _ = find_peaks(p, prominence=prominence)
    valleys, _ = find_peaks(-p, prominence=prominence)
    if not len(peaks):
        return {"status":"not_assessable", "reason":"missing_first_peak", "ratio":None}
    i1 = int(peaks[0])
    after = valleys[valleys > i1]
    if not len(after):
        return {"status":"not_assessable", "reason":"missing_first_valley", "ratio":None}
    i2 = int(after[0])
    points = [i1, i2]
    if zeta_d <= .2:
        after = peaks[peaks > i2]
        if not len(after):
            return {"status":"not_assessable", "reason":"missing_second_peak_for_low_damping_branch", "ratio":None}
        i3 = int(after[0])
        points.append(i3)
        numerator, denominator = p[i1]+p[i3]-2*p[i2], p[i1]+p[i3]+2*p[i2]
        branch = "zeta_le_0p2_three_extrema"
    else:
        numerator, denominator = p[i1]-p[i2], p[i1]+p[i2]
        branch = "zeta_gt_0p2_two_extrema"
    if denominator <= 0 or numerator < 0:
        return {"status":"not_assessable", "reason":"invalid_signed_peak_geometry", "ratio":None}
    return {"status":"ratio_only_not_A116_grade", "ratio":float(numerator/denominator),
            "branch":branch, "extrema": [{"t_s":float(t[i]), "p":float(p[i])} for i in points],
            "A116_grade":None, "A116_missing":"sideslip_phase_and_applicable_test_conditions"}


def settling_with_censoring(time, error, tolerance, *, minimum_tail_s):
    """Settling observed within a finite record, explicitly censored otherwise.

minimum_tail_s and tolerance are experimental choices, not GJB thresholds.
"""
    t, e = np.asarray(time,float), np.asarray(error,float)
    if t.ndim != 1 or e.shape != t.shape or len(t) < 2 or not np.all(np.diff(t)>0) or not np.isfinite(t).all() or not np.isfinite(e).all():
        raise ValueError("Invalid finite record")
    if not math.isfinite(tolerance) or not math.isfinite(minimum_tail_s) or tolerance <= 0 or minimum_tail_s <= 0:
        raise ValueError("Positive tolerance and tail required")
    outside = np.flatnonzero(np.abs(e) > tolerance)
    index = int(outside[-1]+1) if len(outside) else 0
    if index >= len(t) or t[-1]-t[index] < minimum_tail_s:
        return {"settling_time_s":None, "censored":True, "observation_end_s":float(t[-1])}
    return {"settling_time_s":float(t[index]), "censored":False, "observation_end_s":float(t[-1])}
