# MIL-STD-1797A (Notice 3, 2004): lateral-directional coupling and remaining LOES dynamics

Source: `MIL-STD-1797A_notice3_2004.pdf` (849 pp). Text was read from `/tmp/MIL-STD-1797A_notice3_2004.txt`. Graphical boundaries were read from embedded figure images.
Scope: Class IV, Flight Phase Category A, centre stick, unless noted.

**How to read the citations**
- "PDF p.N" is the PDF page index starting at 1, the same as `=====PAGE N=====` in the txt. The document's own printed page number ("doc p.") is sometimes added.
- **The main standard (PDF p.154–172) contains only blanks ("______").** Every number below is an Appendix A "REQUIREMENT GUIDANCE / Recommended values" entry, unless the main-standard text states the number itself.
- Notice 1 pen-and-ink corrections (PDF p.3–12) are already incorporated in the Appendix pages of this reprint, except where noted (see §7.6).
- Flags:
  - **[D]** means the value was digitized from a figure. These are approximate: about ±3–5° in ψβ, about ±0.01–0.02 in ratios, about ±0.3° in Δβ/k.
  - **[G]** means the equation is garbled in the PDF text layer and has been reconstructed. Verify these visually before relying on them.

Page map (PDF p.):

| Item | Main std (blank form) | Appendix A guidance |
|---|---|---|
| Definitions (3.4.6, 3.4.7) | 134–143 | — |
| 4.2.1.1 phugoid | 154 | 297–300 |
| 4.3.1.2 flight-path stability | 159 | 502–504 |
| 4.4.1 / 4.4.1.1 speed stability | 160 | 515–520 |
| 4.5.1.1 T_R / 4.5.1.2 spiral / 4.5.1.3 roll-spiral | 160 | 522 / 538 / 542 |
| 4.5.1.4 roll oscillations | 161 | 550–564 |
| 4.5.1.5 τ_ep | 161 | 565 |
| 4.5.2 roll PIO (Notice-1 replacement) | 161 | 112–113 |
| 4.5.4 lateral accel. at pilot | 161 | 570–573 |
| 4.5.5 roll in steady sideslip | 161 | 574–575 |
| 4.5.8.1 roll performance (for k) | 161–162 | 580–584 |
| 4.5.8.2 / 4.5.8.3 / 4.5.9.5.4 | 162–163 | 608 / 609 / 647 |
| 4.6.1.1 dutch roll | 164 | 651–657 |
| 4.6.1.2 steady sideslips | 164 | 674–676 |
| 4.6.1.3 wings-level turn (DFC) | 164 | 677–679 |
| 4.6.2 yaw due to roll controller | 164 | 686–715 |
| 4.6.6.1 / 4.6.7.1 / 4.6.7.2 | 166–167 | 728 / 733 / 734 |
| 4.7.1 lateral translation | 168 | 745–747 |
| 4.8.1 / 4.8.2 roll coupling, crosstalk | 169 | 748–751 / 752 |
| 4.9.1 / 4.9.2 / 4.9.3 disturbances | 171–172 | 796–802 / 804–827 / 828 |
| App. B equivalent-system matching | — | 829–831 |

---

## 1. Lateral-directional equivalent-system forms and definitions

### 1.1 Transfer-function forms

Notation used in 4.5.1.1: `[ζ,ω] = s²+2ζωs+ω²` and `(1/T) = s+1/T`.

**β/F_rp, the primary dutch-roll fit (4.6.1.1, PDF p.651).** "matching the higher-order sideslip response to yaw control input to the following lower-order form, typically over the frequency range from 0.1 rad/sec to 10 rad/sec":

    β/F_rp = K_β e^(−τ_eβ s) / (s² + 2ζ_d ω_d s + ω_d²)

**Simultaneous φ and β fit (4.6.1.1, PDF p.651).** "Alternatively, when |φ/β|_d is large the φ/F_as equivalent-system transfer function of 4.5.1.4 shall be matched simultaneously with the equivalent system β/F_as transfer function":

    φ/(δ_as or F_as) = A_φ [s² + 2ζ_φ ω_φ s + ω_φ²] e^(−τ_eφ s) / [(s+1/T_S)(s+1/T_R)(s² + 2ζ_d ω_d s + ω_d²)]
    β/(δ_as or F_as) = A_β (s+1/T_β1)(s+1/T_β2)(s+1/T_β3) e^(−τ_eβ s) / [(s+1/T_S)(s+1/T_R)(s² + 2ζ_d ω_d s + ω_d²)]

- Use δ_as for deflection-command controls and F_as for force-command controls.
- The (s+1/T_S) term is the Notice 1 correction (PDF p.9).
- "No limits are set on φ or β numerator terms at this time." (PDF p.651)
- "Limits on τ_eβ have not been specified." (PDF p.652)

**Forms given under roll mode (4.5.1.1, PDF p.522).**

    φ/(δ_as or F_as) = K_φ [ζ_φ, ω_φ] e^(−τ_ep s) / ((1/T_S)(1/T_R)[ζ_d, ω_d])
    β/(δ_rp or F_rp) = (A3 s³ + A2 s² + A1 s + A0) e^(−τ_eβ s) / ((1/T_S)(1/T_R)[ζ_d, ω_d])

- [G] The control subscript of the β form reads "rp" in the text layer.
- Notice 1 (PDF p.7) changed this equation's denominator to "(1/T_S)(1/T_R)[ζ_d,ω_d]" and its numerator to "(A3s³+A2s²+A1s+A0)e^(−τeβ s)".
- Fit range is 0.1–10 rad/s.
- "Where the roll response to F_as contains a significant amount of dutch roll mode, this transfer function should be matched simultaneously with the equivalent-system transfer function of lateral-acceleration response to roll control force. (In general this latter transfer function has a fourth-order numerator…)" (PDF p.522). No explicit a_y/F_as form is given.

**p/F_as (4.5.1.4 guidance, PDF p.550).**

    p/F_as = L_Fas · s [s² + 2ζ_φ ω_φ s + ω_φ²] e^(−τ_ep s) / [(s+1/T_S)(s+1/T_R)(s² + 2ζ_d ω_d s + ω_d²)]

- "When the complex roots cancel (ω_φ = ω_d and ζ_φ = ζ_d), the dutch roll mode is not excited at all."

**r/F_as: NOT FOUND.** No yaw-rate equivalent form is given anywhere.

**Low-order approximations (5.5.1.4, PDF p.563–564) [G, partly].**
- (ω_φ/ω_d)² ≈ 1 − (C′_nδa/C′_lδa)(C′_lβ/C′_nβ)
- ω_d² ≈ g C′_nβ /(C_L1 k_z² b)
- Here C_L1 = W/(q̄S), k_x² = I_x/(mb²), k_z² = I_z/(mb²), and C′ are product-of-inertia-corrected derivatives.
- Adverse yaw: "−180 deg > ψβ > −270 deg ⇔ ω_φ/ω_d < 1".
- Proverse yaw: "0 ≥ ψβ > −90 deg ⇔ ω_φ/ω_d > 1".
- Positive dihedral (p leads β by 45–225°) ⇔ C′_lβ + (ρgb/(4(W/S)))C_yβ C′_lr **negative**.
- Negative dihedral (p leads β by 225° through 360° to 45°) ⇔ the same expression is **positive**.

**Approximation for |φ/β|_d (5.6.2, PDF p.715) [G].**

    |φ/β|_d ≈ { [(L′_β + Y_β L′_r)² + 2ζ_dω_d L′_r (L′_β + Y_β L′_r) + ω_d² L′_r²] / (ω_d² [L′_p² + 2ζ_dω_d L′_p + ω_d²]) }^(1/2)

**Equivalent-system matching (Appendix B, PDF p.829–830).**
- Cost function: (20/n) Σ [(gain_HOS − gain_LOS)² + 0.02 (phase_HOS − phase_LOS)²], with gain in dB and phase in degrees.
- At least 10 frequencies per decade, log-spaced.
- "Modal parameters common to two responses shall be constrained to be identical. For example, the dutch roll mode shall have the same damping and frequency in the roll and sideslip responses."

### 1.2 Symbol definitions (3.4.6 / 3.4.7, PDF p.134–143)

| Symbol | Definition (quoted or paraphrased) | PDF p. |
|---|---|---|
| T_d | Damped dutch-roll period, T_d = 2π/(ω_d√(1−ζ_d²)) | 134 |
| ω_φ, ζ_φ | "Undamped natural frequency / damping ratio of numerator quadratic of φ/F_as transfer function" | 134 |
| p_osc/p_av | "A measure of the ratio of the oscillatory component of roll rate to the average component of roll rate following a yaw-control-free step roll control command". For ζ_d ≤ 0.2: (p1 + p3 − 2p2)/(p1 + p3 + 2p2). For ζ_d > 0.2: (p1 − p2)/(p1 + p2). p1, p2, p3 are roll rates at the 1st, 2nd and 3rd peaks. | 134–135 |
| φ_osc/φ_av | Same formulas with bank angles φ1, φ2, φ3, "following a pedals-free impulse aileron control command" | 135 |
| Δβ_max | "Maximum change in sideslip occurring within 2 seconds or one half-period of the dutch roll, whichever is greater, for a step roll-control command". Notice 1 renamed Δβ to Δβ_max (PDF p.3). "Δβ is the total excursion during the stated time period. Thus if β reverses during that time, the sum of right and left excursion magnitudes is called for" (PDF p.712). | 135, 712 |
| k | Ratio of "commanded roll performance" to the "applicable roll performance requirement" of 4.5.8.1: k = (φ_t)_command/(φ_t)_requirement. Use the **Level 1** φ_t requirement of 4.5.8.1, "with rudder use as permitted by 4.5.8.1" (PDF p.690). Worked example: φ_t req = 30° in 1 s gives k = 9.1/30 = 0.30 (PDF p.142). | 140, 142, 690 |
| t_nβ | Time for the dutch-roll component of the sideslip response to reach its n-th local maximum (right command) or minimum (left command). Measured from half-amplitude of the control input; for pulse inputs, from halfway through the pulse. | 140 |
| ψ_β | "Phase angle expressed as a lag for a cosine representation of the dutch roll oscillation in sideslip": ψ_β = −(360/T_d)·t_nβ + 360(n−1) [deg]. Worked example: −(360/3.5)(2.95) = −303° (PDF p.141). Equivalently β_d ∝ e^(−ζ_dω_d t) cos(ω_d√(1−ζ_d²) t + ψ_β) for a step F_as (PDF p.552) [G]. | 140–141, 552 |
| ∠p/β | "Phase angle between roll rate and sideslip in the free dutch roll oscillation. Angle is positive when p leads β by an angle between 0 and 180 deg" | 140 |
| \|φ/β\|_d | "At any instant, the ratio of amplitudes of the bank-angle and sideslip-angle envelopes in the dutch roll mode" | 140 |
| r̈_o/ṗ_o | "Ratio of initial yawing acceleration to initial rolling acceleration for a step roll control input (equivalent to N′_δas/L′_δas)" | 142 |
| μ | "Rudder shaping parameter: μ = δ_rp(3) − 1", where δ_rp(3) is the pedal deflection 3 s after a step roll input that "would be required for perfect coordination" | 142 |
| δ′_rp(3) | Yaw-to-roll crossfeed parameter (4.6.2) | 134 |

---

## 2. Roll oscillations (4.5.1.4; main std PDF p.161; guidance PDF p.550–557)

The main-standard text reads: "With yaw controls free, the response to roll control commands shall meet the following requirements: ____". The recommended values (PDF p.550) are below.

**(a) First-minimum roll-rate ratio.** "Following a step roll control command, the roll rate at the first minimum following the first peak shall be of the same sign and not less than the following percentage of the roll rate at the first peak":

| Level | Cat A & C | Cat B |
|---|---|---|
| 1 | **60 %** | 25 % |
| 2 | **5 %** | 0 % |

**(b) Roll-rate oscillation.** "For step roll control commands up to the magnitude which causes a 60-degree bank angle change in 1.7 T_d seconds, the value of the parameter p_osc/p_av shall be within the limits shown on figure 156." The figure is on PDF p.551.

**(c) Bank-angle oscillation.** "Following an impulse roll command as abrupt as practical …, the value of the parameter φ_osc/φ_av shall be within the limits shown on figure 157." The figure is on PDF p.551.

**Reading the figures.**
- Both figures have two ψ_β scales:
  - Upper scale, 0 to −360°: "when p leads β by 45° to 225°" (positive dihedral).
  - Lower scale, shifted by 180°: "when p leads β by 225° through 360° to 45°" (negative dihedral).
- The tables below use the upper scale.
- The figures are maximum allowed values.
- Lines between breakpoints are straight, except that the Level-1 rising segment is slightly concave.

**Figure 156, p_osc/p_av, Cat A & C [D]:**

| ψ_β (deg) | 0 → −130 | −130 → −200 | −200 → −270 | −270 → −340 | −340 → −360 |
|---|---|---|---|---|---|
| Level 1 | 0.05 | rises to 0.25 (≈0.14 at −160, ≈0.19 at −180) | 0.25 | falls linearly to 0.05 | 0.05 |

| ψ_β (deg) | 0 → −120 | −120 → −200 | −200 → −270 | −270 → −350 | −350 → −360 |
|---|---|---|---|---|---|
| Level 2 | 0.10 | linear to 0.60 | 0.60 | linear to 0.10 | 0.10 |

Cat B Level 2, for reference [D]: 0.2 up to about −110°, rising to 1.0 at about −180°, 1.0 to about −290°, then falling to about 0.2 at −360°.

**Figure 157, φ_osc/φ_av, Cat A & C [D].** The pattern is shifted about +90° relative to Fig. 156.
- Level 1: 0.05 from 0 to ≈−38°; concave rise to 0.25 at ≈−105° (≈0.12 at −60°, ≈0.18 at −80°); 0.25 to ≈−178°; linear fall to ≈0.05 at ≈−247°; 0.05 to −360°.
- Level 2: 0.10 from 0 to ≈−27°; linear to 0.60 at ≈−104°; 0.60 to ≈−178°; linear to 0.10 at ≈−256°; 0.10 to −360°.

**Interpretation and applicability notes (PDF p.550, 553, 556, 557).**
- "for all Category A operations the suggested Level 1 value of ζ_d in 4.6.1.1 effectively eliminates the need for this requirement".
- The limits for "0 ≥ ψβ ≥ −90 deg are far more stringent than for −180 deg ≥ ψβ > −270 deg".
- "4.5.1.4 need not be applied if |φ/β|_d is small (… < 1.5 generally …)".
- "The requirement is of most value when ζ_d is near the Level 1 boundary (0.1–0.2)".
- If T_S is very small, p_osc/p_av can be misleading (Fig. 160, PDF p.556).

**Related roll-axis LOES limits for Class IV Cat A.**

| Parameter | Level 1 | Level 2 | Level 3 | Source |
|---|---|---|---|---|
| T_R max (s), Table XXIV | 1.0 | 1.4 | 10 † | PDF p.522 |
| Spiral, min time to double (s), Table XXV, Cat A & C | 12 | 8 | 4 | PDF p.538 |
| Coupled roll-spiral, Table XXVI, Cat A | "shall not exhibit a coupled roll-spiral mode" | same | same | PDF p.542 |
| τ_ep max (s), Table XXVII | 0.10 | 0.20 | 0.25 | PDF p.565 |

† Only one "10" is printed in the Level-3 column, which appears to apply to all rows.

- The spiral requirement applies "following a disturbance in bank of up to 20 degrees". Cat B values are 20/8/4 s.
- Roll-spiral damping ζ_RSω_RS for Cat B and C only: 0.5 / 0.3 / 0.15.

**Roll PIO (Notice-1 replacement page, PDF p.112).** "The recommended minimum phase angle of the bank angle frequency response to the roll stick force at the criterion frequency, ω_c, is −180°", where ω_c = (0.24 rad-oct/dB-sec)·S + 6.0 rad/s and S is the average slope of |φ/F_as| in dB/oct over 1–6 rad/s.

---

## 3. Dutch roll (4.6.1.1; main std PDF p.164; guidance PDF p.651–652)

Main-standard text: the requirements "shall be met in trimmed flight with cockpit controls fixed and with them free, and in steady maneuvers, in oscillations of any magnitude …; If the oscillation is nonlinear with amplitude, the requirement shall apply to each cycle of the oscillation."

**Table XL. Recommended minimum dutch roll frequency and damping (PDF p.652):**

| Level | Flight Phase Cat. | Class | min ζ_d* | min ζ_dω_d* (rad/s) | min ω_d (rad/s) |
|---|---|---|---|---|---|
| 1 | A (CO, GA, RR, TF, RC, FF, AS) | I, II, III, IV | **0.4** | **0.4** | **1.0** |
| 1 | A (other) | I, IV | **0.19** | **0.35** | **1.0** |
| 1 | A (other) | II, III | 0.19 | 0.35 | 0.4 |
| 1 | B | All | 0.08 | 0.15 | 0.4 |
| 1 | C | I, II-C, IV | 0.08 | 0.15 | 1.0 |
| 1 | C | II-L, III | 0.08 | 0.10 | 0.4 |
| 2 | All | All | **0.02** | **0.05** | **0.4** |
| 3 | All | All | **0** | — | **0.4** |

\* "The governing damping requirement is that yielding the larger value of ζ_d, except that a ζ_d of 0.7 is the maximum required for Class III."

**The ω_d²|φ/β|_d correction (PDF p.651–652, exact text).** "When ω_d²|φ/β|_d is greater than 20 (rad/sec)², the minimum ζ_dω_d shall be increased above the ζ_dω_d minimums listed in table XL by:

- Level 1: Δζ_dω_d = 0.014(ω_d²|φ/β|_d − 20)
- Level 2: Δζ_dω_d = 0.009(ω_d²|φ/β|_d − 20)
- Level 3: Δζ_dω_d = 0.005(ω_d²|φ/β|_d − 20)

with ω_d in radians per second."

Note that "the Level 1 minimum ζ_d (ζ_d ≥ 0.4) is not supported" by the pilot-rating data. It was retained on AFFTC's recommendation for CO/GA tracking (PDF p.652–653).

---

## 4. Sideslip excursions / yaw response to roll controller (4.6.2; main std PDF p.164; guidance PDF p.686–715)

Main-standard text: "The sideslip excursions to step roll control inputs shall meet the following criteria: ____. Yaw controls shall be free and, in initial steady turns, fixed at the deflection for zero sideslip in the turn."

### Alternative 1 (PDF p.686–687)

**(a) Small inputs.** "The amount of sideslip following a small step roll control command shall be within the limits shown in figure 235 for Levels 1 and 2. This requirement applies for step roll control commands up to the magnitude that causes a 60 degree bank angle change within T_d or 2 seconds, whichever is longer."

Figure 235 plots Δβ/k (deg) against ψ_β, from 0 to −360°, as maximum allowed values [D]:

| Boundary | Low plateau | Ramp | High plateau | Fall |
|---|---|---|---|---|
| **Cat A, Level 1** | **2°** for 0 → ≈−130° | slightly concave rise to 6° at ≈−200° | **6°** for −200 → ≈−270° | linear to 2° at ≈−340°, then 2° to −360° |
| Cat B & C, Level 1 | 3° for 0 → −120° | linear to 10° at −200° | 10° for −200 → −270° | linear to 3° at ≈−350° |
| **All Cat., Level 2** | **4°** for 0 → ≈−110° | linear to 15° at ≈−200° | **15°** for −200 → −270° | linear to ≈4° at −360° |

The breakpoints were deliberately made identical to those of Fig. 156 (PDF p.689). Their meaning (sketch, PDF p.689–690):
- 0 to about −90°: "cross control" is needed to coordinate.
- −180 to −260°: the pilot can coordinate naturally ("Right F_rp for Right Roll").
- −270 to −360°: "coordination becomes increasingly difficult".

**(b) Large inputs (Table XLIII, PDF p.687).** "Following larger step roll control commands, the ratio of the sideslip increment, Δβ, to the parameter k (3.4.6) should be less than the values specified in table XLIII. The roll command shall be held fixed until the bank angle has changed at least 90 degrees."

| Level | Cat. | Adverse (right roll command gives right sideslip) | Proverse (left roll command gives right sideslip) |
|---|---|---|---|
| 1 | **A** | **6°** | **2°** |
| 1 | B and C | 10° | 3° |
| 2 | All | **15°** | **4°** |

Applicability notes:
- Δβ/k "is not as useful as p_osc/p_av when |φ/β|_d > 3.5–5.0" (PDF p.690).
- "For |φ/β|_d above some nominal value (≈5.0), Δβ/k adds little …" (PDF p.692).
- The most important test conditions are those with |φ/β|_d < 6 (PDF p.715).

### Alternative 2, crossfeed criterion (PDF p.700–715; "There is no such alternative requirement in MIL-F-8785C")

The requirement text: "The yaw control crossfeed necessary to maintain zero sideslip shall result in a ratio of initial yawing acceleration to initial rolling acceleration for a step roll control input, r̈_o/ṗ_o, within the limits shown on figure 247 for Levels 1 and 2. In addition, for values of r̈_o/ṗ_o less than 0.07, the crossfeed parameter δ′_rp(3) must be within the limits of table XLIV."

**Table XLIV. Limits on δ′_rp(3) when r̈_o/ṗ_o < 0.07 (PDF p.700):**

| Level | Adverse yaw | Proverse yaw |
|---|---|---|
| 1 | −0.39 | 0.11 |
| 2 | −1.15 | 0.78 |

**Figure 247.** The vertical axis is μ and the horizontal axis is r̈_o/ṗ_o on a log scale. The boundaries are curved; approximate edges of the allowed region [D, ±10–15 %]:

| μ | Level 1 (adverse limit … proverse limit) | Level 2 (adverse … proverse) |
|---|---|---|
| +1 | ≈ −0.21 … (dashed) | ≈ −0.42 … (dashed) |
| 0 | ≈ −0.31 … +0.13 | ≈ −0.54 … +0.31 |
| −1 | ≈ −0.30 … ≈ +0.85 to 1.0 (nose at μ ≈ −1.1) | ≈ −1.3 … ≈ +2.4 |
| −2 | ≈ −0.12 … +0.12 | ≈ −0.67 … ≈ +1.8 |

**Crossfeed model (PDF p.702–706; derivatives in stability axes).**
- Ideal crossfeed: Y_CF = δ_rp/δ_as = −N^β_δas/N^β_δrp.
- Its first-order LOS form is Y_CF ≈ N′_δas[s+1/T_βas] / (N′_δrp[s+1/T_βrp]).
- μ = (T_βrp/T_βas) − 1.
- Required pedal for a unit step: δ_rp(t) = (N′_δas/N′_δrp)[1 + μ(1 − e^(−t/T_βrp))].
- Hence δ_rp(∞)/δ_rp(0+) = 1 + μ, and μ ≈ δ_rp(3) − 1.
- Physical meaning of μ (Table XLVII, PDF p.708):
  - μ = 0: N′_p = g/U₀ (pure-gain crossfeed).
  - −1 < μ < 0: N′_δas and N′_p are opposing.
  - μ < −1: rudder reversals are needed.

**Table XLV ground rules (PDF p.701):**
- T_R, ω_d and ζ_d are Level 1.
- |φ/β|_d < 1.5 when turbulence is a factor and N′_δas/L′_δas > 0.03.
- Meets Level 2 p_osc/p_av.

---

## 5. Other yaw/roll cross-coupling and steady-sideslip requirements

### 5.1 Roll in steady sideslip, "dihedral effect" (4.5.5, main std PDF p.161 states it in full; guidance PDF p.574)

"In the straight, steady sideslips of 4.6.1.2:
- a. An increase in right bank angle, or no change, shall accompany an increase in right sideslip, and an increase in left bank angle, or no change, shall accompany an increase in left sideslip.
- b. Right or zero roll-control deflection and force shall accompany right sideslips, and left or zero roll-control deflection and force shall accompany left sideslips. For Levels 1 and 2, the variation of roll-control deflection and force with sideslip angle shall be essentially linear."

Waveoff exception: "no more than 50 percent of roll-control power … and no more than 10 pounds of roll-control force" in the opposite direction.

Guidance notes:
- "This requirement normally (but not necessarily) will preclude negative effective dihedral."
- "It is not desirable to use rudder-to-aileron crossfeeds to meet this requirement inasmuch as such mechanizations augment L_δrp and not L_β".
- Fighter pilots "desired some dihedral to enable them to damp the dutch roll using ailerons alone".

**Upper limits on positive dihedral:**

| Requirement | Recommended value | Source |
|---|---|---|
| 4.5.8.2, roll power in steady sideslip | "no more than **75 percent** of roll power should be required" | PDF p.608 |
| 4.5.9.5.4, roll force in steady sideslip | max **10 lb** roll-stick force (20 lb wheel). Final approach: Level 1 10 lb, Levels 2–3 20 lb, at the sideslip of 4.6.6.1 (10°). | PDF p.647 |
| 4.5.8.3(c), roll power with sideslip in power approach | "For sideslip angles of **10 deg** in the power approach, not more than 75 percent of available control power" | PDF p.609 |

### 5.2 Steady sideslips, roll due to yaw control and pedal linearity (4.6.1.2; main std PDF p.164; guidance PDF p.674)

- Applies to "yaw-and-roll-control-induced steady, upright, zero-yaw-rate sideslips with the aircraft trimmed for wings-level straight flight, at sideslip angles up to those produced or limited by: a. Full yaw-control-pedal deflection, or b. 250 pounds of yaw-control-pedal force, or c. Maximum roll control or surface deflection".
- "Right yaw-control-pedal force and deflection shall produce left sideslips, and left … shall produce right sideslips."
- "For larger sideslip angles, an increase in yaw-control-pedal force shall always be required for an increase in sideslip."
- **Recommended linear ranges:** "±15 deg for β/δ_rp, ±10 deg for β/F_rp. Essentially linear … is qualitative rather than absolute." (PDF p.674)
- The steady-sideslip gradients dβ/dδ_r, dδ_a/dβ and dφ/dβ in terms of the C-derivatives are given on PDF p.575 and p.675 [G].
- Pedal force gradients: "The quantitative rudder pedal force requirements of this Standard are only on maximum forces, not on gradients" (4.6.7, PDF p.732).

### 5.3 Turn coordination and pedal forces

**4.6.7.1, pedal force to hold zero sideslip in the 4.5.8.1 rolls (PDF p.733):**

| Class | Cat. | Level | Max pedal force (lb) |
|---|---|---|---|
| IV | A | 1 | **50** |
| IV | A | 2–3 | 100 |
| IV | B, C | All | 100 |
| All others | All | All | 100 |

**4.6.7.2, steady coordinated turns (PDF p.734).** Class IV: **60° bank with ≤ 40 lb pedal force**, trimmed for wings-level flight (Levels 1 and 2). Classes I and II: 45°/40 lb. Class III: 30°/40 lb.

**4.5.8.1, roll performance needed to compute k.** Table XXX, Class IV (PDF p.583). Pedals are free for Class IV Level 1 (main std PDF p.162). Cat A, Level 1:

| Speed range | Bank change | Time (s) |
|---|---|---|
| VL | 30° | 1.1 |
| L | 30° | 1.1 |
| M | 90° | 1.3 |
| H | 50° | 1.1 |

- Cat A, Level 2: VL 30°/1.6 s, L 30°/1.5 s, M 90°/1.7 s, H 50°/1.3 s.
- Speed ranges (PDF p.582): VL is V_min ≤ V < V_min+20 kt; L is up to 1.4V_min; M is up to 0.7V_max; H is up to V_max.
- CO and GA phases have their own tables, XXXI–XXXIII.

### 5.4 Lateral acceleration and side force

**4.5.4, lateral acceleration at the pilot station (PDF p.570).** "the ratio of maximum lateral acceleration at the pilot station to maximum roll rate shall not exceed ___ for the first 2-1/2 seconds following a step roll control input" (yaw control free, and also used to coordinate turn entry). Recommended values of n_y,pilot,max/p_max:

| Level | Limit (g/(deg/s)) |
|---|---|
| 1 | 0.012 |
| 2 | 0.035 |
| 3 | 0.058 |

- The guidance says "Some refinement of the limits may be necessary for Class I, II, and IV".
- Pilot-station lateral acceleration (PDF p.572): a_y = v̇ + ru − pw − g cosθ sinφ − z ṗ + x ṙ.

**Side-force / DFC modes:**
- 4.6.1.3 wings-level turn (PDF p.678), Table XLI bandwidth of heading or lateral path angle to the DFC input:
  - Tracking (Cat A): Level 1 1.25 rad/s, Level 2 0.60 rad/s.
  - Path deviation (Cat A, C): Level 1 0.30 rad/s, Level 2 0.12 rad/s.
  - Bandwidth is the lesser of the 45° phase-margin frequency and the 6 dB gain-margin frequency (PDF p.679).
- 4.7.1 lateral translation (PDF p.745–746): "at least **4 deg of sideslip**"; no bandwidth value ("no data of sufficient quality"); "0.5g is about the maximum tolerable lateral acceleration without special restraints".

### 5.5 Other related items

- **4.1.11.8 control cross-coupling (PDF p.151), qualitative only:** "No controller shall create a secondary response which is objectionable (for Levels 1 and 2) or dangerous (for Level 3)."
- **4.6.6.1(d), minimum sideslip in power approach from yaw control power:** 10° (PDF p.728).
- **4.6.4, crosswind (Table XLIX, PDF p.717):** Class IV Levels 1–2, 30 kt; Level 3, half of that.

---

## 6. Long-term pitch / speed / flight-path

**4.2.1.1, long-term pitch response (main std PDF p.154; guidance PDF p.297).**
- The requirement applies to "Any oscillation with a period of 15 seconds or longer".
- "no aperiodic flight path divergence is allowed within the Service Flight Envelope for any Level" (except as provided in 4.4.1, 4.4.1.1 and 4.2.1.2).
- It applies with controls fixed and free.
- Recommended values:
  - Level 1: equivalent ζ_p > 0.04
  - Level 2: equivalent ζ_p > 0
  - Level 3: T_2 > 55 s
- Level 3 is checked from the time response "for both nose-up and nose-down control impulses".

**Long-period LOES form (PDF p.297).** The text calls this the "three-degree-of-freedom equivalent classical longitudinal response":

    θ/(δ_es or F_es) = K_θ (s+1/T_θ1)(s+1/T_θ2) e^(−τ_e s) / ([s² + 2ζ_p ω_p s + ω_p²][s² + 2ζ_sp ω_sp s + ω_sp²])

The e^(−τ_e s) term is the Notice 1 correction.

**Phugoid approximations (PDF p.298) [G].** Valid only subsonic:
- ω_p² ≈ 2(g/V)² · (static margin / manoeuvre margin), i.e. (N − c.g.)/(N_m − c.g.) = h_n/h_m.
- 2ζ_pω_p ≈ 2(g/V)(C_D/C_L).

**Flight-path stability, dγ/dV (4.3.1.2; guidance PDF p.502).** This is defined only for the **landing approach**; there is no Cat A requirement.
- "the curve of flight-path angle versus true airspeed shall have a local slope at V_o min that is negative or less positive than":

| Level | Max slope dγ/dV (deg/kt) |
|---|---|
| 1 | 0.06 |
| 2 | 0.15 |
| 3 | 0.24 |

- Thrust is set for the normal glide path at V_o min.
- "The slope … at 5 knots slower than V_o min shall not be more than 0.05 degrees per knot more positive than the slope at V_o min."
- Related relations (PDF p.503–504) [G]:
  - h/δe = (As³ + Bs² + Cs + D)/([s² + 2ζ_pω_p s + ω_p²][s² + 2ζ_spω_sp s + ω_sp²]).
  - 1/T_h1 ≈ D/C, with C ≈ V(Z_δe M_w − M_δe Z_w).
  - dγ/dδe|ss = (1/V)·D/(ω_p²ω_sp²).

**Speed stability (4.4.1; main std PDF p.160; guidance PDF p.515).**
- For Levels 1 and 2 the main-standard text says there shall be "no tendency for the airspeed to diverge aperiodically when the aircraft pitch attitude is disturbed from trim".
- Recommendation (i), applying "Over an equilibrium speed range of 15 percent about trim or 50 kt, whichever is less":
  - "a. For rapid commanded pitch attitude changes the short-term airspeed change shall be in the same direction as the final value."
  - "b. For a fixed positive change in commanded pitch attitude from trim, the steady-state airspeed shall not increase."
- Recommendation (ii), Level 3: "time for the airspeed perturbation to double amplitude … shall not be less than 6 seconds".
- Airspeed/attitude numerator ratio (PDF p.516) [G]. For X_δ = 0, u/θ = N^u_δ/N^θ_δ, with zero 1/T_u1 over zeros 1/T_θ1 and 1/T_θ2. It is garbled and not reproduced.
- 4.4.1.1 transonic relaxation (PDF p.520), centre stick, 1 g:
  - Levels 1–2: local gradient no more unstable than 3 lb per 0.01 M, and no more than 10 lb in the unstable direction.
  - Level 3: 6 lb per 0.01 M and 20 lb.

---

## 7. Atmospheric disturbance model (4.9; main std PDF p.171–172; guidance PDF p.796–828)

### 7.1 Allowable degradation (Table LII, PDF p.797, Normal States)

| Disturbance | Within Operational Flight Envelope | Within Service Flight Envelope |
|---|---|---|
| Light to calm | Level 1, Satisfactory | Level 2, Acceptable |
| Moderate to light | Level 1, Acceptable | Level 2, Controllable |
| Severe to moderate | Controllable | Recoverable |

Table LIII (PDF p.798) covers Failure States I and II. The verification guidance recommends that the main simulation effort go into the "Moderate" level (PDF p.803).

### 7.2 Spectra (4.9.2, PDF p.804)

"Where feasible, the von Karman form shall be used … When no comparable structural analysis is performed or when it is not feasible …, use of the Dryden form will be permissible. In general, both the continuous turbulence model and the discrete gust model shall be used."

Ω is the spatial frequency (rad/ft), and Φ(Ω) = V·φ(ω) (PDF p.142–143).

**Dryden** (footnote 14: "the MIL-F-8785C spectra corrected for a factor of two"):

    Φ_ug(Ω) = σ_u² (2L_u/π) · 1/(1 + (L_u Ω)²)
    Φ_vg(Ω) = σ_v² (2L_v/π) · (1 + 12(L_v Ω)²)/[1 + 4(L_v Ω)²]²
    Φ_wg(Ω) = σ_w² (2L_w/π) · (1 + 12(L_w Ω)²)/[1 + 4(L_w Ω)²]²

**von Karman:**

    Φ_ug(Ω) = σ_u² (2L_u/π) · 1/[1 + (1.339 L_u Ω)²]^(5/6)
    Φ_vg(Ω) = σ_v² (2L_v/π) · [1 + (8/3)(2.678 L_v Ω)²]/[1 + (2.678 L_v Ω)²]^(11/6)
    Φ_wg(Ω) = σ_w² (2L_w/π) · [1 + (8/3)(2.678 L_w Ω)²]/[1 + (2.678 L_w Ω)²]^(11/6)

Note that L_v and L_w here are the "half" scales, L_u = 2L_v = 2L_w.

**Table LIV, digital filter (shaping-filter) implementation for Dryden (PDF p.806) [G, rotated table].** T is the time step and η_i are independent unit-noise sequences with rms σ_η.

| Component | Intensity | Break frequency | Finite-difference equation |
|---|---|---|---|
| u_g | σ_u = σ | a_u = V/L_u | u_g = (1−a_uT)u_g + (σ_u/σ_η)√(2a_uT)·η1 |
| v_g | σ_v = σ | a_v = V/L_u | v_g = (1−a_vT)v_g + (σ_v/σ_η)√(2a_vT)·η2 |
| w_g | σ_w = σ (printed "σ_v = σ") | a_w = V/L_u | w_g = (1−a_wT)w_g + (σ_w/σ_η)√(2a_wT)·η3 |
| p_g* | σ_p = 1.9σ_w/√(2L_w b) | a_p = 2.6/√(2L_w b) (as printed; dimensionally needs ×V) | p_g = (1−a_pT)p_g + (σ_p/σ_η)√(2a_pT)·η4 |
| q_g** | — | a_q = πV/(4b) | q_g = (1−a_qT)q_g + (π/(4b))(w_g − w_g,PAST) |
| r_g† | — | a_r = πV/(3b) | r_g = (1−a_rT)r_g + (π/(3b))(v_g − v_g,PAST) |

Footnotes as extracted [G]:
- \* p_g is significant only if (b/2L_w)·C_lp > C_lβ.
- \*\* q_g is significant only if (πb/16L_w)·C_mq > C_mα.
- † r_g is significant only if (πb/12L_v)·C_nr > C_nβ.
- Note that all three translational filters are printed as first-order with break frequency V/L_u.

**Continuous spectra Φ_pg, Φ_qg, Φ_rg: NOT GIVEN.** They are listed only as symbols (PDF p.143). Only the Table LIV filters are given.

**Discrete gust (PDF p.805).** "1 − cosine": v = (v_m/2)(1 − cos(πx/d_m)) for 0 ≤ x ≤ d_m, and v_m for x > d_m.
- d_m is tuned to each natural frequency.
- Light and Moderate magnitudes come from Fig. 265 (PDF p.809, **not digitized**).
- Severe magnitudes: 66 ft/s EAS at V_G, 50 at V_o max, 25 at V_max, reduced linearly above 20,000 ft (PDF p.808).

### 7.3 Medium/high altitude (PDF p.805–808, Fig. 262 on p.799)

- "turbulence above 2,000 feet is isotropic": σ_u = σ_v = σ_w.
- **Dryden: L_u = 2L_v = 2L_w = 1,750 ft. von Karman: 2,500 ft.**
- RMS intensity from the simplified (solid) lines of Fig. 262 [D] (ft/s TAS, all axes):
  - **Light:** 5 up to ≈9,000 ft, linear to 3 at ≈17,000–17,500 ft, then 3.
  - **Moderate:** 10 up to ≈11,000 ft, linear to 3 at ≈45,000 ft, then 3.
  - **Severe:** ≈21 (21.2) from ≈4,000 ft to 20,000 ft, linear to 3 at ≈80,000 ft. Below ≈4,000 ft it slants down to ≈15.5 at ≈2,000 ft.
- "a minimum rms magnitude of 3 ft/sec is specified at all altitudes" (PDF p.798).
- The high/low split is "above approximately 1750 ft" in 4.9.1 (PDF p.172 and p.796) but "approximately 2000 ft" in the guidance (PDF p.798, 805).

### 7.4 Low altitude (PDF p.800, 810–811; Figs. 263, 266, 267)

- σ_w = 0.1 u₂₀. Here u₂₀ is the mean wind at 20 ft, read from Fig. 263 [D]:

| Intensity | u₂₀ | Approx. probability of exceedance |
|---|---|---|
| Light | ≈15 kt | ≈10⁻¹ |
| Moderate | ≈30 kt | ≈10⁻³ |
| Severe | ≈45 kt | ≈10⁻⁵ |

- Fig. 266 scales: L_u = 2L_v = h/(0.177 + 0.000823h)^1.2 and 2L_w = h, for 10 < h < 1000 ft.
- Fig. 267 intensities: σ_u/σ_w = σ_v/σ_w = 1/(0.177 + 0.000823h)^0.4 below 1000 ft, and 1.0 above.
- Mean-wind shear profile: u_w = u₂₀ · ln(h/z₀)/ln(20/z₀), with z₀ = 0.15 ft for Cat C and 2.0 ft for other phases.
- Vector shear:
  - Light: 0°.
  - Moderate: 90° over 600 ft.
  - Severe: 90° over 300 ft.
- Steady crosswind at 20 ft (PDF p.800): Light 0–10 kt, Moderate 11–30 kt, Severe 31–45 kt.
- Wind-shear capability:
  - Decreasing headwind: gγ_max up to 3.4 ft/s².
  - Increasing headwind: gγ_min up to 1.7 ft/s².
  - Vector shear: 9 deg/s at 20 kt.
  - Duration: at least 10 s.
- The carrier airwake model is also given (PDF p.812 onward; not extracted).

### 7.5 How gusts enter the equations (4.9.3, main std PDF p.172 and guidance PDF p.828; Table LVII, PDF p.825)

- "The gust and turbulence velocities shall be applied to the aircraft equations of motion through the aerodynamic terms only, and the direct effect on the aerodynamic sensors shall be included when such sensors are part of the aircraft augmentation system."
- "The uniform immersion is accounted for by u_g, v_g, and w_g defined at the aircraft center of gravity. The angular velocities due to turbulence are equivalent in effect to aircraft angular velocities. Approximations … (precise only at very low frequencies)":
  - Appendix p.828 and Notice-1 correction (PDF p.12) [G]: −α̇_g = q_g = ∂w_g/∂x, p_g = ∂w_g/∂y, r_g = −∂v_g/∂x.
  - Main standard p.172, uncorrected [G]: −α̇_g = q_g = ∂w_g/∂x, **p_g = −∂w_g/∂y**, r_g = −∂v_g/∂x.
- "For altitudes below 1750 ft, … u_g aligned along the horizontal relative mean wind vector and w_g vertical."
- "it is acceptable to consider u_g and v_g as being one-dimensional …. The w_g should be considered two-dimensional, a function of both x and y" when structural modes matter.
- Table LVII sets the gust-derivative signs:
  - Column headers: (−q_g) over ∂w_g/∂x, (−r_g)₁ over ∂u_g/∂y, (−r_g)₂ over ∂v_g/∂x, (p_g) over ∂w_g/∂y.
  - ΣX: −X_u u_g, −X_w w_g, −X_v v_g.
  - ΣZ: −Z_u u_g, −Z_w w_g, plus small Z_ẇ, Z_q and Z_v terms.
  - ΣM: −M_u u_g, −M_w w_g, −M_ẇ ẇ_g, M_q ∂w_g/∂x, −M_v v_g.
  - ΣY: −Y_v v_g (Y_v̇ and Y_r terms are small).
  - ΣL: −L_v v_g, −L_p p_g.
  - ΣN: −N_v v_g, −N_v̇ v̇_g, −N_r(∂v_g/∂x), −N_p p_g.
  - This equals evaluating the aero forces on (u−u_g, v−v_g, w−w_g, p−p_g, q−q_g, r−r_g).
- **An explicit "α_g = w_g/V, β_g = v_g/V" statement was NOT FOUND.** It is only implied by "aerodynamic terms only" and Table LVII.

### 7.6 Sign-convention caution

The minus signs in the 4.9.3 equation are positioned ambiguously in the text layer, and the q_g sign differs between 4.9.3 and Table LVII. Check PDF p.12, p.172 and p.828 visually before fixing the convention in code.

---

## 8. Inertial / kinematic roll coupling (4.8.1; main std PDF p.169; guidance PDF p.748–751)

**Requirement text:** "In yaw-control-free maximum-performance rolls through ___ degrees, entered from straight flight or during turns, pushovers, or pullups ranging from 0 g to 0.8 n_L, including simultaneous pitch and roll commands, none of the resulting yaw or pitch motions, or sideslip or angle of attack changes, shall exceed structural limits or cause other dangerous flight conditions such as uncontrollable motions or roll autorotation. Rudder pedal inputs used to roll the aircraft …, shall not result in departures in pitch, roll, or yaw. During combat-type maneuvers involving rolls through angles up to ___ degrees and rolls which are checked at any given bank angle …, the yawing and pitching shall not be so severe as to impair the tactical effectiveness of the maneuver. These requirements define Level 1 and 2 operation."

- **Recommended value for both blanks: Class I and IV, 360°.** Classes II and III: 120°. "but not to exceed the structural roll angle limit applicable to the normal load factor" (PDF p.748).
- Pitch inertial-coupling term (PDF p.748): q̇ = ((I_z − I_x)/I_y)·pr + (I_xz/I_y)(r² − p²) + M_aero/I_y. Also "M = −I_xz p²" when rolling about the flight path rather than the principal axis (PDF p.749).
- "inertial pitch/roll coupling may set the pitch control power requirements …, especially … Class IV" (PDF p.748).
- A stability-axis yaw damper reduces sideslip but increases the pr and p² terms (PDF p.748–749).
- Verification (PDF p.750): "At least a 5-degree-of-freedom nonlinear analysis (constant speed is generally a good assumption) is needed."
- A quadratic for the critical roll rate p_cr² in terms of ω_sp², ω_d², k_x, k_y and k_z is given (PDF p.750–751) but is **[G] badly garbled and not reproduced**. Its uncoupled approximations, ω_sp² and ω_d² ≈ (g/(C_L1 k_z² b))·C_nβ + …, are on PDF p.751.
- **4.8.2, pitch/roll crosstalk (PDF p.752):** "5 percent cross-coupling has been suggested by one source as 'normal and not objectionable'", but the guidance does not propose any hard limit.
- 4.8.4.x (high-α, departure) was not extracted.

---

## Not found, or only partly readable

1. No **r/F_as** (or a_y/F_as) equivalent-system form is given. No limits are placed on the φ or β numerators, or on τ_eβ (PDF p.651–652).
2. The continuous **Φ_pg, Φ_qg, Φ_rg** spectra are not given; only the Table LIV discrete filters are. The Table LIV text is rotated and garbled:
   - the p_g break frequency is printed without V;
   - the w_g intensity is printed as "σ_v";
   - the √ signs are inferred.
3. The **α_g = w_g/V, β_g = v_g/V** statement is not explicit.
4. **Figure-only boundaries** were digitized and are approximate: Figs. 156, 157, 235, 247, 262 and 263. Fig. 265 (discrete-gust magnitudes) was **not digitized**.
5. These equations are garbled in the text layer and reconstructed or omitted:
   - 4.9.3 gust-rate sign convention (main standard and appendix disagree on p_g);
   - |φ/β|_d approximation (PDF p.715);
   - phugoid approximations (PDF p.298);
   - critical roll-rate quadratic (PDF p.750);
   - steady-sideslip gradient formulas (PDF p.575, 675).
6. Flight-path stability dγ/dV is specified **only for the landing approach** (Cat C). There is no Cat A requirement, and no quantitative speed-stability gradient for Levels 1–2 beyond "no aperiodic divergence" and the 4.4.1 recommendations.
