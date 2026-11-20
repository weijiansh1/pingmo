# Notes: Davidson & Andrisani, "Lateral-Directional Eigenvector Flying Qualities Guidelines for High Performance Aircraft", NASA TM-110306, Dec 1996

File: `NASA_19970008392_lateral_eigenvector.pdf`. Page numbers below are **PDF page numbers**. The printed page number is the PDF page minus 2 (for example, PDF p.4 is printed page 2). The PDF is a scan, so I read every equation below from the page images, not from the garbled text extraction.

Notation used here: K = N'_δ/L'_δ (the report calls it the "adverse yaw ratio"), Φ = (φ/β)_dr, B = (β/p)_roll, λ_r = λ_roll (= -1/T_R), λ_s = λ_sprl, λ_dr = -ζ_dr ω_dr + j ω_dr sqrt(1-ζ_dr²). δ = δ_stk = δ_ail, which corresponds to F_as in our model.

---

## 1. Model and axis system

- Linearized rigid-body lateral-directional equations for steady, straight, level flight, **stability axes** (McRuer, Ashkenas & Graham 1973) (PDF p.4, eq. 2.1a/b).
- State vector x = {β, p, r, φ}^T. Inputs are {δ_ail, δ_rud} (PDF p.4, eq. 2.1b; also eq. 3.42 on PDF p.26). Here p and r are stability-axis rates and φ is bank angle. Primes mean the inertia (I_xz) terms are included: L'_i = L_i + (I_xz/I_x) N_i (PDF p.40, eq. A.1). The kinematics are p = sφ, with no θ/tanθ terms.
- State matrix rows (PDF p.4):
  - β̇ = [Y_β β + (Y_p+α_0) p + (Y_r−1) r + (g/V_0) φ + Y_δ δ]/(1−Y_β̇)
  - ṗ = L'_β β + L'_p p + L'_r r + L'_δ δ
  - ṙ = N'_β β + N'_p p + N'_r r + N'_δ δ
  - φ̇ = p
- Characteristic polynomial: Δ(s) = k_Δ (s−λ_s)(s−λ_r)(s²+2ζ_dr ω_dr s+ω_dr²), with k_Δ = Y_β̇ − 1 (PDF p.5, eqs. 2.2–2.3).
- The **simplifying assumptions** used throughout:
  - (2.12a) Y_r ≈ 0, Y_β̇ ≈ 0, Y_δ ≈ 0. Here "Y_r ≈ 0" means the (Y_r − 1) term reduces to −1.
  - (2.12b) |Y_β(N'_r L'_δ − N'_δ L'_r)| << |N'_β L'_δ − N'_δ L'_β| (PDF p.6).
  - For β/δ only: (2.18a) Y_p + α_0 ≈ 0, (2.18b) (g/V_0)[N'_δ L'_r − L'_δ N'_r] ≈ 0, and the spiral pole is near the origin (PDF p.7).
  - Additional approximations in the eigenvector work: Y_β L'_r << L'_β, (g/V_0) L'_r << ω_dr², and (g/V_0) L'_r/λ_r << L'_p (PDF p.19, p.21).
- Modal approximations (PDF p.5):
  - ω_dr² ≈ N'_β (2.4)
  - 2ζ_dr ω_dr ≈ −N'_r − Y_β − (L'_β/N'_β)(N'_p − g/V_0) (2.5)
  - λ_roll ≈ L'_p − (L'_β/N'_β)(N'_p − g/V_0) (2.6)
  - λ_sprl ≈ (g/(λ_roll V_0)) (N'_r L'_β/N'_β − L'_r) (2.7)
- **Single input only.** The analysis uses lateral stick driving the aileron with zero rudder (PDF p.6). Pedal-input transfer functions are **not derived**. Pedal appears only indirectly: K can be changed by blending lateral and directional effectors, for example through an aileron-rudder interconnect (PDF p.19).

## 2. Transfer functions, eigenvectors and residues

### 2.1 Roll channel (this matches our fixed p/F_as form)

- φ/δ = L'_δ (s²+2ζ_φ ω_φ s+ω_φ²) / [(s−λ_s)(s−λ_r)(s²+2ζ_dr ω_dr s+ω_dr²)] (2.9, PDF p.6)
- p/δ = s·φ/δ (2.15, PDF p.7). This is the same structure as our p/F_as, with L_Fa ↔ L'_δ.
- Exact numerator coefficients (eqs. 2.10–2.11, PDF p.6). Under assumptions (2.12) they reduce to (PDF p.7):
  - ω_φ² ≈ N'_β [1 − (N'_δ/L'_δ)(L'_β/N'_β)] (2.13)
  - 2ζ_φ ω_φ ≈ −N'_r − Y_β + (N'_δ/L'_δ) L'_r (2.14)
- Steady-state roll rate for a unit step input, with the spiral pole near 0: p_ss = L'_δ (−1/λ_roll)(ω_φ/ω_dr)² (2.16, PDF p.7)

### 2.2 Sideslip channel

- Full numerator: eq. 2.17 (PDF p.7).
- With (2.12a) and (2.18), one factor of s cancels the spiral pole near the origin (2.19, PDF p.7; repeated as 3.27, PDF p.24):

  β/δ = −N'_δ [ s − L'_p + (L'_δ/N'_δ)(N'_p − g/V_0) ] / [(s−λ_r)(s²+2ζ_dr ω_dr s+ω_dr²)]

  which can be rewritten as (3.28, PDF p.24):

  β/δ = −L'_δ [ K(s − L'_p) + (N'_p − g/V_0) ] / [(s−λ_r)(s²+2ζ_dr ω_dr s+ω_dr²)]

- r/δ is given in eqs. 2.20–2.21 (PDF p.8).

### 2.3 Modal residues and input participation (PDF p.9–10)

- Output: y_i(t) = Σ_j Σ_k R_{i,j,k} ∫ e^{λ_j(t−τ)} u_k(τ) dτ, with **R_{i,j,k} = c_i v_j l_j b_k** (2.34). Here v_j is the right eigenvector, l_j is the j-th row of V⁻¹ (the left eigenvector), and b_k is the k-th column of B.
- Step response: y_i = Σ (R_{i,j,k}/λ_j) e^{λ_j t} (2.36).
- In the lateral form, the Dutch-roll term is 2|R/λ_dr| e^{−ζω t} cos(ω_dr sqrt(1−ζ²) t + Ψ). The phases are **Ψ_p = ∠(R_{p,dr}/λ_dr), Ψ_β = ∠(R_{β,dr}/λ_dr), Ψ_r = ∠(R_{r,dr}/λ_dr)** (2.37–2.39, PDF p.10). The constant offset β_0 is given by (2.38).
- "Eigenvalues determine time constant/frequency/damping; eigenvectors determine the residues" (PDF p.10).
- The report never bounds the input-participation term l_j b_k directly. Input participation enters only through the control-derivative ratio **N'_δ/L'_δ**. Control derivatives "are elements of the B matrix and are therefore independent of system eigenvectors", while the stability derivatives can be related to the eigenvectors (PDF p.19).

### 2.4 Eigenvector element ratios from cofactors (PDF p.18–19)

- (x_i/x_j)_mode k = (−1)^(i−j) [Δ_qi(s)/Δ_qj(s)] evaluated at s = λ_k, for any q (3.2).
- (φ/β) = −Δ32/Δ31 = [−L'_r(s−Y_β) + L'_β]/[s² − L'_p s − (g/V_0) L'_r] (3.5)
  - At λ_dr: **(φ/β)_dr ≈ (−L'_r λ_dr + L'_β)/(λ_dr² − L'_p λ_dr)** (3.6, PDF p.20). Its phase is eq. 3.7, and eq. 3.8 is the corresponding solution for L'_r.
- ∠(φ/β)_dr = ∠(p/β)_dr − ∠λ_dr, with ∠λ_dr = 180° − acos ζ_dr (3.9–3.10).
  - Positive dihedral ⇔ 45° < ∠(p/β)_dr < 225° (Chalk et al. 1969).
  - Therefore −135° + acos ζ_dr < ∠(φ/β)_dr < 45° + acos ζ_dr (3.11, PDF p.20).
- **Design choice ∠(φ/β)_dr = 0.** Costigan-Calico pilots preferred it, it guarantees positive dihedral for any stable DR pole, and it gives (PDF p.20–21):
  - **L'_r ≈ −(L'_β/N'_β)(L'_p + 2ζ_dr ω_dr)** (3.12)
  - L'_p ≈ λ_roll + (L'_β/N'_β)(N'_p − g/V_0) (3.13)
  - L'_r ≈ −(L'_β/N'_β)[λ_roll + (L'_β/N'_β)(N'_p − g/V_0) + 2ζ_dr ω_dr] (3.14)
- **Roll-mode sideslip content** (PDF p.21):
  - (β/p) = −Δ31/Δ32 = [−(g/(V_0 s)) L'_r + (s − L'_p)]/[L'_β − L'_r(s−Y_β)] (3.15)
  - (β/p)_roll ≈ (λ_roll − L'_p)/(L'_β − λ_roll L'_r) (3.16)
  - (β/p)_roll ≈ −(L'_β/N'_β)(N'_p − g/V_0)/(L'_β − λ_roll L'_r) (3.17)
  - Eq. 3.18 is the same ratio after substituting 3.14 and ω_dr² ≈ N'_β.
- Inverting for (N'_p − g/V_0) (3.19, PDF p.22):

  (N'_p − g/V_0) ≈ −B (λ_r² + 2λ_r ζ_dr ω_dr + ω_dr²) / [1 + λ_r B (L'_β/N'_β)]

- **Dihedral ↔ DR mode shape** (3.20–3.21, PDF p.22). The bracketed term in 3.20 is exactly −1, which gives:

  **(φ/β)_dr = |φ/β|_dr ≈ −L'_β/N'_β** (positive dihedral, ∠ = 0)

  Meeker-Hall give a more general approximation (2.53, PDF p.15):

  |φ/β|_dr ≈ |L'_β/N'_β| · sqrt[(1 + N'_β L'_r²/L'_β²)/(1 + L'_p²/N'_β)]

### 2.5 KEY RESULT: p/δ numerator zeros in terms of eigenvectors and K

- **ω_φ² ≈ ω_dr² [1 + K (φ/β)_dr]** (3.22, PDF p.22; repeated as 3.51, PDF p.29)
- **2ζ_φ ω_φ ≈ 2ζ_dr ω_dr + Φ { K(λ_r + 2ζ_dr ω_dr) + B (λ_r² + 2λ_r ζ_dr ω_dr + ω_dr²)(1 + ΦK) / (1 − λ_r B Φ) }** (3.26, PDF p.23; repeated as 3.52, PDF p.29). Intermediate forms are 3.23–3.25 (PDF p.22–23).
- The derivation requires 1 + K|φ/β|_dr > 0 (PDF p.32, p.34) and complex zeros, i.e. −1 < ζ̃_φ < 1 and ω̃_φ > 0 (PDF p.34).
- When |φ/β|_dr = 0, the zeros do not depend on B or K. As |φ/β|_dr grows, the zeros become more sensitive to both (PDF p.24).
- At B = 0 and K = 0, the zeros cancel the DR poles exactly and p_osc/p_avg = 0 (PDF p.35).

### 2.6 KEY RESULT: β/δ in terms of eigenvectors and K

(3.29–3.31, PDF p.24–25)

  **β/δ ≈ −L'_δ [ K(s − λ_r) − (1 + KΦ) · B (λ_r² + 2λ_r ζ_dr ω_dr + ω_dr²) / (1 − λ_r B Φ) ] / [(s−λ_r)(s²+2ζ_dr ω_dr s+ω_dr²)]** (3.31)

Residues (PDF p.26–31):

- p residues: R_{p,j} = L_δ λ_j (λ_j − z_φ)(λ_j − z̄_φ) / Π_{m≠j}(λ_j − λ_m) (3.47–3.49).
  - R_{p,dr} vanishes when z_φ = λ_dr.
  - z̃_φ comes from 3.50–3.52, and the approximate residues are 3.54–3.56.
- β residues, with spiral eigenvector (β/φ)_sprl = 0 so that R_{β,sprl} = 0 (3.61–3.64, PDF p.31):
  - R_{β,roll} = −L'_δ[K(λ_r − L'_p) + (N'_p − g/V_0)] / [(λ_r − λ_dr)(λ_r − λ̄_dr)] (3.63)
  - R_{β,dr} = −L'_δ[K(λ_dr − L'_p) + (N'_p − g/V_0)] / [(λ_dr − λ_r)(λ_dr − λ̄_dr)] (3.64)
- Eigenvector form (PDF p.32):
  - **R̃_{β,roll} = L'_δ (1 + KΦ) B / (1 − λ_r B Φ)** (3.67)
  - R̃_{β,dr} = −L'_δ [K + (λ_r − λ̄_dr)(1 + KΦ)B/(1 − λ_r B Φ)] / (λ_dr − λ̄_dr) (3.68)
  - So the roll-mode content of β/F_as is directly proportional to (β/p)_roll.

### 2.7 Kinematic and side-force constraints on the remaining elements

(PDF p.25–26)

- (p/φ)_dr = λ_dr, (p/φ)_roll = λ_roll, (p/φ)_sprl = λ_sprl (3.33)
- (r/β)_dr = −λ_dr + Y_β + (g/V_0)(φ/β)_dr ≈ −λ_dr (3.35–3.36)
- (r/p)_roll ≈ −(β/p)_roll λ_roll + g/(V_0 λ_roll) (3.39)
- (r/φ)_sprl ≈ g/V_0 (3.41)

### 2.8 Derived consequence for our fixed roll channel

**This is my own derivation, NOT stated in the report.** It combines eqs. 3.22, 3.26 and 3.31 and holds only under the report's assumptions: single aileron input, ∠(φ/β)_dr = 0, Y-terms ≈ 0, and spiral pole/zero cancellation.

Let A = ω_φ²/ω_dr² and Q = λ_r² + 2λ_r ζ_dr ω_dr + ω_dr² = |λ_r − λ_dr|². Then:

- K = (A − 1)/|φ/β|_dr
- B·|φ/β|_dr = D / (Q·A + D·λ_r), where D = 2ζ_φω_φ − 2ζ_dω_d − (A−1)(λ_r + 2ζ_dω_d)
- β/δ ≈ −(L'_δ/|φ/β|_dr) · [ (A − 1) s + 2A ζ_dr ω_dr − 2ζ_φ ω_φ ] / [(s − λ_r)(s² + 2ζ_dr ω_dr s + ω_dr²)]

So once p/F_as and the poles are fixed, **|φ/β|_dr is the only free eigenvector parameter**. It scales β/F_as by 1/|φ/β|_dr and sets the required aileron yaw K.

Check: I built derivative sets from eqs. 3.13, 3.14, 3.19 and 3.21. Numerically, the exact 4-state model reproduced the s² and s coefficients of the β numerator. The constant term is the one the report drops via (2.18b). If ∠(φ/β)_dr ≠ 0, the identity picks up an extra term proportional to [L'_r + (L'_β/N'_β)(L'_p + 2ζ_dω_d)].

## 3. Guidelines: what is constrained, with numbers

### 3.1 Desired eigenvectors

Two elements per eigenvector can be assigned: 2 controls and full-state feedback (PDF p.27).

| Mode | Desired eigenvector over (β, p, r, φ) | Specified element and reason | Constrained elements | Source |
|---|---|---|---|---|
| Spiral | [0 * * 1]^T | (β/φ)_sprl = 0, giving R_{β,sprl} = 0 and coordinated banking/turning | p/φ = λ_s, r/φ ≈ g/V_0 | PDF p.27, eq. 3.43 |
| Roll | [(β/p)_roll 1 * *]^T | (β/p)_roll; it "effects the cancellation of the Dutch roll mode in the roll response" | φ/p = 1/λ_r, r/p from eq. 3.39 | PDF p.27, eq. 3.44 |
| Dutch roll | [(1,0) (*,*) (*,*) (\|φ/β\|_dr, 0°)]^T, entries given as (magnitude, phase) | \|φ/β\|_dr and ∠(φ/β)_dr = 0 | r/β ≈ −λ_dr, p/β = (φ/β)λ_dr | PDF p.28, eq. 3.45 |

The design space of the guidelines is therefore (|φ/β|_dr, (β/p)_roll, N'_δ/L'_δ), with the eigenvalues fixed (PDF p.28, p.34).

### 3.2 Military Standard criteria the guidelines come from

The guidelines translate MIL-STD-1797A §4.5.1.4 (Roll Oscillations) and §4.6.2 (Yaw Axis Response to Roll Controller) (PDF p.13). Both are based on Meeker & Hall 1967 and Chalk et al. 1969 (AFFDL-TR-69-72).

**Roll-rate oscillation, p_osc/p_avg** (PDF p.15–16; Fig. 4, PDF p.44)

- Definition: p_osc/p_avg = (p1 + p3 − 2p2)/(p1 + p3 + 2p2) for ζ_dr ≤ 0.2, and (p1 − p2)/(p1 + p2) for ζ_dr > 0.2 (2.54–2.55).
- Level 1 limit, as a function of Ψ_β (positive dihedral):
  - 0.05 for 0 ≥ Ψ_β ≥ −130° and for −340° ≥ Ψ_β ≥ −360°
  - 0.25 for −200° ≥ Ψ_β ≥ −270°
  - linear between those regions
- The bank angle change must always be in the commanded direction.
- The requirement applies to step commands up to the size that gives 60° bank in 1.7 T_d, where T_d = 2π/(ω_dr sqrt(1−ζ_dr²)) (2.57).

**Sideslip excursion, Δβ_max/k_β** (PDF p.16–17; Fig. 5, PDF p.45)

- Δβ_max = max β − min β over 0 < t < t_β, where t_β = max(2 s, T_d/2).
- k_β = φ(t_req)/φ_req; for fighters, k_β = φ(1 s)/60° (2.58–2.60).
- Level 1 limit: 2 for 0 ≥ Ψ_β ≥ −130° and −340° ≥ Ψ_β ≥ −360°; 6 for −200° ≥ Ψ_β ≥ −270°; linear between. Units are presumably degrees; the report does not say explicitly.
- The requirement applies up to 60° bank in max(T_d, 2 s).

**Phase link** (PDF p.15–16, p.33)

- Ψ_1 is the angular position of the p/δ zero relative to the DR pole (Fig. 3, PDF p.43): 0° = zero to the left, 90° = below, 180° = right, 270° = above.
- Ψ_1 = atan[(ω_dr sqrt(1−ζ_dr²) − ω_φ sqrt(1−ζ_φ²)) / (−ζ_dr ω_dr + ζ_φ ω_φ)] (3.75).
  - It is probably meant as a four-quadrant atan; the report does not say. **UNCERTAIN.**
- Ψ_β ≈ Ψ_1 − 270° for positive dihedral (2.56/3.76).
  - Chalk et al. say this is "relatively independent of roll and spiral eigenvalue locations".
  - The approximation Ψ̃_β uses ω̃_φ, ζ̃_φ from 3.51–3.52 (3.77, PDF p.33).

**Which criterion to use** (PDF p.15, p.34)

| \|φ/β\|_dr | Where DR contamination shows up | Criterion |
|---|---|---|
| < ~1.5 | yaw/sideslip | sideslip excursion |
| > ~5 (PDF p.15 says 3.5–5) | roll rate | roll-rate oscillation |
| in between | both | both must be met |

Meeker-Hall also found that |φ/β|_dr ≳ 12 combined with light DR damping was unsatisfactory (PDF p.14).

### 3.3 Example guidelines

All examples use λ_s = −0.005, λ_r = −2.5, ω_dr = 2.0 rad/s, ζ_dr = 0.1 (PDF p.35). The Level 1 regions are closed, tilted, elongated regions around (B, K) = (0, 0). The bounds below are **approximate readings from the figures**.

| \|φ/β\|_dr | Criterion | (β/p)_roll range | N'_δ/L'_δ range | Figure |
|---|---|---|---|---|
| 10 | roll-rate oscillation | ≈ −0.009 to +0.023 | ≈ −0.032 to +0.032 | Fig. 10, PDF p.49 |
| 5 | roll-rate oscillation | ≈ −0.017 to +0.047 | ≈ −0.065 to +0.065 | Figs. 11, 13, PDF p.50, 52 |
| 5 | sideslip excursion | ≈ −0.03 to +0.046 | ≈ −0.10 to +0.07 | Figs. 12, 13, PDF p.51–52 |
| 1 | sideslip excursion | ≈ −0.032 to +0.025 | ≈ −0.12 to +0.055 | Fig. 14, PDF p.53 |

- In every case the region slopes so that larger (β/p)_roll pairs with larger N'_δ/L'_δ.
- For |φ/β|_dr = 5, the roll-oscillation guideline is the more restrictive one over most of the boundary (PDF p.36).
- The roll-oscillation guideline shrinks strongly as |φ/β|_dr increases. The sideslip-excursion guideline is fairly insensitive to |φ/β|_dr (PDF p.36–37).
- Fig. 7 (PDF p.47), |φ/β|_dr = 10: the p_osc/p_avg = 0.05 contour is a small ellipse with B ≈ ±0.005 and K ≈ ±0.013. The 0.25 contour reaches B ≈ 0.035 and K ≈ 0.15.
- Fig. 8 (PDF p.47): Ψ̃_β contours radiate from (0, 0). The lenient sector (−200° to −270°) lies to the lower right, i.e. B > 0 with K below the −270° line.
- Fig. 6 (PDF p.46), |φ/β|_dr = 5: ω_φ = 1.5, 2, 2.5 and 3 rad/s at K ≈ −0.0875, 0, 0.1125 and 0.25. This matches eq. 3.22. ζ_φ = 0.4 passes through (B ≈ 0.038, K = 0).
- Text inconsistency: the prose often writes "(p/β)_roll", but the equations and all figure axes use **(β/p)_roll**. I treat (β/p)_roll as correct.

## 4. ω_φ/ω_d, aileron yaw, dihedral, adverse vs proverse

- ω_φ²/ω_dr² ≈ 1 − (N'_δ/L'_δ)(L'_β/N'_β) (2.13/3.3, PDF p.7, p.19). Equivalently, ω_φ²/ω_dr² ≈ 1 + (N'_δ/L'_δ)(φ/β)_dr (3.22, PDF p.22).
- The location of the p/δ zeros relative to the DR poles depends mainly on the control coupling N'_δ/L'_δ and the stability coupling terms L'_β/N'_β, (N'_p − g/V_0) and L'_r (PDF p.19).
- Sign convention: "A negative value of L_β is positive dihedral" (PDF p.40). So (φ/β)_dr = −L'_β/N'_β > 0 for positive dihedral.
- **Inference, not stated explicitly:** with positive dihedral, K < 0 gives ω_φ < ω_dr, and K > 0 gives ω_φ > ω_dr.
  - With the usual convention L'_δ > 0, K < 0 is adverse aileron yaw.
  - The Meeker-Hall tables agree: every configuration with ω_φ/ω_dr < 1 has N'_δ/L'_δ < 0 (PDF p.54–56).
  - The report calls N'_δ/L'_δ the "adverse yaw ratio" (PDF p.23, 25, 28) but never states its sign explicitly.
- The appendix explains that aileron yaw is adverse (opposite to the turn) because of differential induced drag, and that N_p is a component of adverse yaw (PDF p.40).
- Sideslip phasing (PDF p.17):
  - When β is mainly **adverse** (out of the turn being rolled into), pilots tolerate a lot of sideslip.
  - When β is mainly **proverse** (into the turn), pilots tolerate only a little, because coordination is difficult.
  - This is why the Level 1 limits are lenient for −200° ≥ Ψ_β ≥ −270°.
- Meeker-Hall (PDF p.14, Fig. 1 on PDF p.41):
  - The best ratings came when the p/δ zeros roughly cancelled the DR poles.
  - Zeros to the left of the DR pole rated better than zeros to the right.
  - Lower-left-quadrant zeros degraded the least (Ψ_1 ≈ 0–90°, i.e. Ψ_β ≈ −270° to −180°).
  - The best-rated cases in the tables have ω_φ/ω_dr ≈ 0.93–0.97, i.e. slightly adverse. This is my reading of the tables.

## 5. Test cases and tabulated values (usable as realistic ranges)

### Meeker-Hall 1967 (variable-stability T-33, fighter mission)

58 configurations, Cooper ratings (PDF p.14; Tables 1–7, PDF p.54–57).

**Modal data**

| Group | ω_dr (rad/s) | ζ_dr | \|φ/β\|_dr | ∠(φ/β)_dr | τ_roll (s) | τ_spiral (s) |
|---|---|---|---|---|---|---|
| AB | 2.36–2.55 | 0.10–0.12 | 1.34–1.59 | 44–49° | 0.4 | −91, 995, ∞ |
| BB | 2.49–2.60 | 0.08–0.10 | 4.82–5.22 | 42–47° | 0.37–0.4 | 166–997 |
| CB | 2.44–2.48 | 0.08–0.10 | 12.0–14.7 | 39–47° | 0.4–0.5 | −250, 111, −104 |

Real aircraft therefore sit near ∠(φ/β)_dr ≈ 40–50°, not 0°.

**Zero and control data** (Tables 2, 4, 6)

- ω_φ/ω_dr ranges from 0.23 to 1.49, ζ_φ from 0.03 to 0.25, and N'_δ/L'_δ from −0.166 to +0.183.
- L'_δ is 54–239 (deg/s²)/in. N'_p is −0.39 to +0.211 1/s.
- CB-1a, CB-2a and CB-3a have real zeros, marked * in the table. The meaning of the listed pair is **UNCERTAIN**.
- Ratings for the best cases:
  - AB-2: PR 3 at ω_φ/ω_dr = 0.93–1.02 (K = −0.038 to +0.047).
  - BB-2c: PR 2 at 0.94 (K = 0).
  - BB-3d: PR 2 at 0.97 (K = 0.037).
  - CB: best PR is 4.5, and several configurations rate 9–10.

**Stability derivatives** (Table 7)

| Quantity | Range |
|---|---|
| g/V_0 | 0.0547 1/s |
| L'_β | −10.6 to −13.3 (AB), −40.6 to −45.1 (BB), −106 to −116 (CB) 1/s² |
| L'_p | −2.2 to −2.9 1/s |
| L'_r | −0.06 to 13.96 1/s |
| N'_β | 3.06 to 6.01 1/s² |
| N'_r | +0.013 to −0.71 1/s |
| Y_β | −0.13 to −0.16 1/s |
| Y_p + α_0 | ≈ 0.02 |
| Y_r − 1 | ≈ −0.997 |

### Costigan & Calico 1989 (YA-7D, eigenspace-assignment control laws)

(PDF p.17; Tables 8–10, PDF p.58)

- Test matrix: |φ/β|_dr ∈ {0, 1.5, 3.0} × ∠(φ/β)_dr ∈ {0°, 60°, 120°}, 7 configurations.
- Design parameters (Table 9):
  - Roll: λ = −4.0, eigenvector (β, p, r) = (0, 1, 0).
  - Spiral: λ = −0.025, eigenvector (β, φ) = (0, 1).
  - DR: ω = 3.0 rad/s, ζ = 0.4.
- Results:
  - Pilot ratings varied little with |φ/β|_dr.
  - Pilots preferred ∠ = 0°. Yaw-pointing CH = 4.2–6.2 at 0° versus 5.7–6.0 at 120°. Bank-tracking CH = 3.8–4.2 at 0° versus 4.5–4.7 at 120°.
  - The authors blamed the poor YA-7D stick dynamics for the small rating differences.

### Report's own guideline examples

λ_s = −0.005, λ_r = −2.5, ω_dr = 2.0 rad/s, ζ_dr = 0.1, with |φ/β|_dr = 10, 5 and 1 (PDF p.35).

## Uncertainties and caveats

- All figure-derived bounds in §3.3 are visual readings, roughly ±0.003 in B and ±0.01 in K.
- The report's derivations are approximate (2.12, 2.18) and all assume ∠(φ/β)_dr = 0.
- The pedal (F_rp) channels are not covered by this report.
- §2.8 is my own derivation, not the report's.
