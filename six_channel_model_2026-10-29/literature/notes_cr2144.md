# NASA CR-2144 (Heffley & Jewell, 1972) "Aircraft Handling Qualities Data" -- extraction notes

Files produced (this directory):
- `cr2144_lateral.csv` -- 94 rows (10 aircraft x all flight conditions), lateral-directional derivatives, mass/inertia, TF factors, HQ parameters, plus computed columns.
- `cr2144_longitudinal.csv` -- the same 94 rows, longitudinal derivatives and TF factors, plus computed stability-axis quantities.

Rows: NT-33A 8, F-104A 10, F-4C 10, X-15 10, HL-10 10, Jetstar 10, CV-880M 7, B-747 10, C-5A 9, XB-70A 10 = 94.
Every value in the "as tabulated" columns was read visually from the scanned printout (OCR layer not used for numbers).
Bare airframe / SAS off, bobweight loop open, in every case.

## 1. Axis system, units, definitions (Appendix A, B, C of the report)

- **All tabulated dimensional derivatives, inertias, TF factors and HQ parameters are BODY AXIS** (x along the fuselage reference line, FRL), as stated in every table header ("BODY AXIS SYSTEM"). The non-dimensional aero tables in the report are partly stability-axis, but those were not used.
- Lateral rolling/yawing derivatives are **primed (inertia-coupled)**: L'_x = (L_x + Ixz N_x/Ix) G, N'_x = (N_x + Ixz L_x/Iz) G, G = 1/(1 - Ixz^2/(Ix Iz)) (Appendix A p. A-18). Ix, Iy, Iz, Ixz are body (FRL) axis, slug-ft^2 (CV-880M tables give Ixz = 0 and epsilon = 0; XB-70A Ixz is negative).
- Units: ft, s, rad, lb. Y_v [1/s] = (rho S V/2m) Cy_beta; Y_beta = V_T0 * Y_v [ft/s^2]; L'_beta, N'_beta [1/s^2]; L'_p, L'_r, N'_p, N'_r [1/s]; Y*_delta = (rho S V/2m) Cy_delta [1/s per rad]; L'_delta, N'_delta [1/s^2 per rad].
- Longitudinal (body axis): X_u*, Z_u* [1/s] include the thrust Mach derivative (X_u* = X_u + T_u cos xi); M_u* [1/(ft s)]; X_w, Z_w [1/s]; M_w [1/(ft s)]; Z_wdot [-]; Z_q [ft/s]; M_wdot [1/ft]; M_q [1/s]; X_delta, Z_delta [ft/s^2/rad]; M_delta [1/s^2/rad]. Report definitions: M_alpha = U0 M_w, M_alphadot = U0 M_wdot (I add Z_alpha = U0 Z_w as a derived column). U0 = V cos(alpha0), W0 = V sin(alpha0), theta0 = alpha0 + gamma0.
- Report lateral EOM (Appendix C, used for its TFs): beta_dot = Y_v beta + (W0/V) p - (U0/V) r + (g cos(theta0)/V) phi + Y* delta; p_dot = L'_beta beta + L'_p p + L'_r r + L' delta; r_dot = N'... ; phi_dot = p + r tan(theta0). (Y_p, Y_r not used.)
- TF factor notation: (s + 1/T) and (s^2 + 2 zeta omega s + omega^2); a printed value in parentheses means the opposite (a "Z/W" printed in parentheses are two real roots 1/T; a "1/T" pair printed in parentheses are zeta;omega). Lateral denominator: 1/T(DET)1 = spiral 1/T_s, 1/T(DET)2 = roll 1/T_R, Z/W(DET)1 = dutch roll. Longitudinal: Z/W(DET)1 = phugoid, Z/W(DET)2 = short period; when the phugoid splits into real roots they are given in `phugoid_real_roots_1overT_tab` (1/T values; negative = unstable) and zeta_p/omega_p are left empty. N(PHI/DA) gives zeta_phi, omega_phi. 1/T_theta1, 1/T_theta2 are the theta/delta_e numerator zeros.
- HQ parameters (body axis): W(PHI)/W(D) = omega_phi/omega_d; PHI TO BETA = |phi/beta| of the dutch-roll mode (rad/rad) and its phase (deg, printed modulo 360; e.g. 390 = 30).

## 2. Sign conventions (Appendix A symbols, p. A-5)
- delta_a positive for positive (right-wing-down) rolling moment -> L'_delta_a > 0 in all rows.
- delta_r positive for **nose-LEFT** yawing moment (negative N) -> N'_delta_r < 0 and Y*_delta_r > 0 in all rows (X-15 uses the all-moving vertical tail delta_v with the same convention).
- delta_e (and stabilizer delta_s for F-104A, F-4C, X-15; elevon for XB-70A) positive trailing-edge-down / nose-down -> M_delta < 0.
- p positive right wing down, r positive nose right, beta positive with v out the right wing.
- With these conventions a "proverse/adverse" check: N'_delta_a/L'_delta_a > 0 is proverse aileron yaw, < 0 adverse.

## 3. X_w / X_alpha and the trim-lift term (coordinator question)
Yes -- the tabulated body-axis X_w contains the trim-lift (lift-vector rotation) contribution. Appendix A/B define X_w = (rho S U0/2m)[-C_X_alpha - 2(W0/U0)(C_X + M/2 C_X_M)] with C_X the body axial-force coefficient (positive aft) and, from Appendix B, C_X_alpha = C_D_alpha cos a0 - C_D sin a0 - C_L_alpha sin a0 - **C_L cos a0**. Hence X_alpha (stability axes) = (qS/m)(C_L - C_D_alpha) ~= g - D_alpha/m in level flight. In the longitudinal CSV I rotated the body-axis X/Z u,w derivatives to stability axes exactly with Appendix B (F_s = R^T F_b R; M_u,s = M_u c + M_w s), and give X_alpha_stab = V X_w,stab and the aerodynamic part X_alpha_aero = X_alpha_stab - g cos(gamma0) (~ -(qS/m) C_D_alpha; approximate because it assumes L ~= W cos gamma and ignores thrust-vector alpha dependence). X_u_stab*V/g ~= -2/(L/D) only for subsonic, Mach-independent drag/thrust (X_u* contains T_u and transonic C_D_M terms), so the implied L/D is listed only for M <= 0.8.

## 4. Computed columns and spot checks
TF factors are tabulated for every row, so no factor had to be synthesized; nevertheless I recomputed them from the tabulated derivatives (numpy/scipy) as a reading check:
- `*_comp` (lateral): the report's full body-axis 4-state model above (with alpha0, theta0).
- `*_s4` (lateral): **the requested simple 4-state model** -- state [beta,p,r,phi], beta_dot = Y_v beta - r + (g/V) phi, phi_dot = p, Y_p=Y_r=0 -- using the primed derivatives rotated exactly into stability axes by alpha0 (columns `*_stab`: L'_s = c L'_b + s N'_b, N'_s = -s L'_b + c N'_b, rate matrix R M R^T).
- `*_comp` (longitudinal): Appendix C 4th-order body-axis model (X_q = 0, X_udot etc. neglected as in the report).
- Agreement (median / 90th percentile relative error over 94 rows), full lateral model: omega_d 0.11%/0.3%, zeta_d 0.17%/0.6%, 1/T_R 0.11%/0.4%, zeta_phi 0.13%/0.3%, omega_phi 0.09%/0.3%, |phi/beta|_d 0.13%/0.4%, 1/T_s 0.4%/2% (spiral is ill-conditioned). Simple 4-state model: identical to <0.5% for all level-flight rows; for the HL-10 (gamma = -35..+14 deg) its 1/T_s is 13-31% low for the glides (+20% for the FC10 climb) and |phi/beta| up to 12% low (because it uses g/V and phi_dot = p instead of g cos(theta0)/V and phi_dot = p + r tan(theta0)). Longitudinal: omega_sp, zeta_sp, omega_p, 1/T_theta1,2 median 0.12-0.14%; zeta_p median 0.5% but lightly-damped phugoids are ill-conditioned (several 5-15% low, NT-33A FC5 -0.029 vs tabulated -0.0078).
- Spot check example (NT-33A FC1, PA, SL M 0.204): tabulated 1/T_s=0.0318, 1/T_R=2.20, zeta_d=0.0609, omega_d=1.13, zeta_phi=0.195, omega_phi=0.848, |phi/beta|=2.14; full model 0.0319, 2.19, 0.0607, 1.13, 0.195, 0.848, 2.15; simple 4-state (stability axes) 0.0319, 2.19, 0.0607, 1.13, 0.195, 0.848, 2.15. beta/delta_r numerator (0.0295)(s-0.0494)(s+2.05)(s+42.3) reproduced (Appendix A example).
- The rudder derivatives were double-checked against the rudder-TF gains A(beta)=Y*_dr, A(p)=L'_dr, A(r)=N'_dr and the beta/delta_r numerator zeros (columns `A_beta_dr_tab`, `beta_dr_num_factors_tab`); aileron derivatives against A(phi), zeta/omega_phi and the beta/delta_a numerator.
- Many single-glyph ambiguities (the printer's "9" looks like "S/5", "0" like "C") were resolved this way; see Section 6.

## 5. Pages read (PDF page = printed page + 4)
Index: PDF 6 (Table I-1). Appendix A/B/C: PDF 321-348. Flight-condition figures: PDF 11, 38, 67, 114, 172, 199, 216, 249, 279.
| aircraft | dim/mass | long. derivs | pitch TF | lat. derivs | aileron TF | rudder TF | lat. HQ |
|---|---|---|---|---|---|---|---|
| NT-33A | 26 | 27 | 28 | 31 | 32 | 33 | 34 |
| F-104A | 53 | 54 | 55 | 60 | 61 | 62 | 63 |
| F-4C | 86 | 87 | 88 | 103 | 104 | 105 | 110 |
| X-15 | 129 | 130 | 131 | 134 | 135 | 136 | 139 |
| HL-10 | 156 | 157 | 158 | 161 | 162 | 163 | 168 |
| Jetstar | 187 | 188 | 189 | 192 | 193 | 194 | 195 |
| CV-880M | 204 | 205 | 206 | 209 | 210 | 211 | 212 |
| B-747 | 233 | 234 | 235 | 238 | 239 | 240 | 245 |
| C-5A | 265 | 266 | 267 | 272 | 273 | 274 | 275 |
| XB-70A | 296 | 297 | 298 | 312 | 313 | 314 | 319 |
The `pdf_page` column lists the PDF pages used for that row; `doc_page` the printed page numbers.
Configurations: FC1 is power approach (PA) for NT-33A, F-104A, F-4C, Jetstar, C-5A, XB-70A; CV-880M and B-747 FC1 = landing (L), FC2 = PA; HL-10 FC1-9 unpowered glides (gamma<0), FC10 powered climb; all other rows nominal (cruise) configuration.

## 6. Uncertain readings, corrections and source inconsistencies
- **NT-33A FC1**: MU* sign mark unclear (read +); computed phugoid zeta 0.093 vs tab 0.0948
- **NT-33A FC2**: MW sign faint, read negative (short period reproduced); phugoid zeta sensitive (computed 0.014 vs tab 0.0199)
- **NT-33A FC5**: alpha printed 'S.40' read 9.40 and zeta_d printed '.C0638' read 0.00638 (both confirmed by recomputing the lateral denominator); computed phugoid zeta -0.029 vs tab -0.0078 (ill-conditioned)
- **NT-33A FC7**: 1/T_s printed '00333' (decimal point missing) read 0.00333 (confirmed by recomputation); phugoid = two real roots
- **F-104A FC3**: SOURCE INCONSISTENCY: tabulated 1/T_theta2=2.29 and phugoid 0.767/0.0523 are not reproduced by the tabulated derivatives (computed 1.79 and 0.863/0.0462); short period agrees
- **F-104A FC4**: phi/beta phase printed 390 deg (=30 deg mod 360)
- **F-104A FC8**: SOURCE INCONSISTENCY: tabulated lateral denominator (1/T_R 1.50, zeta_d 0.0339, 1/T_s 0.00368) and beta/da numerator not reproduced by tabulated derivatives (computed 1/T_R 1.40, zeta_d 0.047, 1/T_s 0.0041); readings double-checked
- **F-4C FC4**: MW printed '-.017?' (5 vs 9 glyph) read -0.0179 (only this reproduces sp 0.308/4.24); 1/T_s glyph 3/8 ambiguous, read 0.00848 (reproduced); XW '.00?71' (2 or 3) read 0.00371 and ZWD '-.00?10' read -0.00210 (both insensitive)
- **F-4C FC6**: L'_da printed '0.00' (field overflow); A(phi)=10.0 and recomputation imply L'_da ~ 10.0
- **F-4C FC7**: ZU '.01?5' read 0.0105 (reproduces phugoid 0.191/0.045)
- **F-4C FC9**: L'_beta glyph '-?.67' (8 vs 9) read -8.67: only this reproduces 1/T_s, zeta_d, |phi/beta|=0.948 and phase 16.9
- **HL-10 FC1**: YV printed '-.3?4' read -0.394 (= YB/VT)
- **HL-10 FC4**: MW printed '-.01?2' read -0.0102 (only this reproduces sp 0.145/2.56)
- **Jetstar FC4**: ZW printed '-1.6?' read -1.69 (5 vs 9 glyph; -1.65 fits slightly worse); XW '.10?' partly obscured read 0.103
- **Jetstar FC7**: MW printed -.0154 with a handwritten mark below; the tabulated TF factors (sp 0.382/3.77, phugoid real roots -0.0195/0.0339) are reproduced only with MW=-0.0164 (value kept as printed)
- **CV-880M FC1**: Iz printed .359E+7 in Table VIII-3 but Fig. VIII-1 lists 4,070,000 for the PA/L configuration
- **CV-880M FC5**: alpha printed '?.30' (6 vs 8) read 8.30 (only 8.30 reproduces the tabulated lateral denominator)
- **B-747 FC1**: Table IX-3 inertias for FC1-2 (.142E8/.323E8/.454E8/870050) differ from Fig. IX-1 PA values (13.7/30.5/43.1/0.825 x1e6); table values kept
- **B-747 FC7**: phugoid tab 0.323/0.00984 vs computed 0.303/0.0102 (ill-conditioned)
- **C-5A FC4**: L'_dr glyph in Table X-9 ambiguous (3 vs 5); 0.500 used: matches rudder-TF gain A(p/dr)=.500 and reproduces beta/dr and phi/dr numerators (0.300 does not)
- **C-5A FC9**: L'_beta printed +0.333 (positive; confirmed by recomputation)
- **XB-70A FC2**: L'_beta printed +9.67 (positive; confirmed by recomputation)
- **XB-70A FC8**: pitch TF column faded: omega_p, A_theta, 1/T_theta1 completed from recomputation consistent with visible digits
- **XB-70A FC9**: pitch TF column faded: omega_p, zeta_sp, omega_sp, 1/T_theta1, 1/T_theta2 completed from recomputation consistent with visible digits
- **XB-70A FC10**: pitch TF column faded: phugoid real roots, omega_sp, 1/T_theta1 completed from recomputation consistent with visible digits
- **NT-33A FC3** rudder beta/delta_r zero printed '-.C0728' = -0.00728.
- **X-15 FC2, FC5** rudder beta/delta_r high-frequency zero printed '6?.1', '??.8' = 69.1, 99.8 (confirmed by recomputation).
- **XB-70A**: Table XI-2 prints Iy = .100E+8 for FC2-10 while Fig. XI-1 gives 19.9e6; the table value violates the rigid-body triangle inequality (Iz=22.1e6 > Ix+Iy=11.8e6), so it is almost certainly a typo. The CSV keeps the printed value; the inertia-ratio summary is given both raw and with 19.9e6.
- **CV-880M FC3-7** inertias give (Iz-Ix)/Iy = 1.03 (slight triangle-inequality violation in the source).
- Inertias in the tables generally agree with the flight-condition figures (F-104A table values are the body-axis transform of the figure's principal-axis values); exceptions noted above (B-747 PA, CV-880M PA, XB-70A).
- XB-70A Table XI-3 also lists "DES" control derivatives; only DE was used. F-104A/F-4C/X-15 pitch control is the stabilizer (DS), tabulated in the DE columns.

## 7. Summary ranges (min / p10 / median / p90 / max over all rows)
Ratios with zeta_d in the denominator explode where zeta_d ~ 0 (Jetstar FC8 zeta_d = 0.0035 -> zeta_phi/zeta_d = 15.6; F-104A FC1 zeta_d < 0 -> negative ratio); use the percentiles.

### Lateral-directional (body axes, as tabulated)
| quantity | unit | n | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|---|
| Y_v (=Y_beta/V) | 1/s | 94 | -0.791 | -0.317 | -0.126 | -0.0565 | -0.0352 |
| L'_r | 1/s | 94 | -1.01 | -0.0557 | 0.315 | 0.897 | 7.17 |
| N'_p | 1/s | 94 | -0.341 | -0.15 | -0.0381 | 0.0146 | 0.145 |
| N'_da / L'_da | - | 94 | -0.11 | -0.0511 | 0.0428 | 0.123 | 0.379 |
| L'_dr / N'_dr | - | 94 | -6.61 | -4.05 | -0.911 | -0.306 | 3.34 |
| Y*_dr / N'_dr | s | 94 | -0.0995 | -0.0496 | -0.0156 | -0.00472 | -0.00198 |
| omega_phi/omega_d (TF factors) | - | 94 | 0.428 | 0.728 | 0.955 | 1.08 | 1.3 |
| zeta_phi/zeta_d (TF factors) | - | 94 | -2.93 | 0.8 | 1.15 | 3.03 | 15.6 |
| |phi/beta|_d | rad/rad | 94 | 0.0391 | 0.884 | 1.98 | 3.83 | 6.12 |

### Same quantities after exact rotation of the primed derivatives to stability axes (by alpha0)
| quantity | unit | n | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|---|
| L'_r (stab) | 1/s | 94 | -0.693 | -0.00801 | 0.347 | 1 | 7.27 |
| N'_p (stab) | 1/s | 94 | -0.237 | -0.0621 | -0.00398 | 0.105 | 0.462 |
| N'_da / L'_da (stab) | - | 94 | -0.286 | -0.177 | -0.0332 | 0.0418 | 0.389 |
| L'_dr / N'_dr (stab) | - | 94 | -5.91 | -2.32 | -0.712 | -0.187 | 3.99 |
| Y*_dr / N'_dr (stab) | s | 94 | -0.0974 | -0.0483 | -0.0149 | -0.00422 | -0.00204 |

### Additional lateral context (tabulated)
| quantity | unit | n | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|---|
| zeta_d | - | 94 | -0.0345 | 0.0278 | 0.084 | 0.176 | 0.266 |
| omega_d | rad/s | 94 | 0.53 | 0.811 | 1.83 | 4.51 | 7.53 |
| 1/T_R | 1/s | 94 | 0.124 | 0.402 | 1.06 | 2.45 | 7.86 |
| 1/T_s | 1/s | 94 | -0.0264 | -0.00206 | 0.00788 | 0.0352 | 0.0868 |
| L'_beta | 1/s^2 | 94 | -363 | -68.1 | -6.04 | -0.782 | 16.6 |
| N'_beta | 1/s^2 | 94 | 0.106 | 0.452 | 2.41 | 13.5 | 42.7 |
| L'_p | 1/s | 94 | -7.36 | -2.46 | -1.08 | -0.461 | -0.175 |
| N'_r | 1/s | 94 | -1.2 | -0.486 | -0.205 | -0.101 | -0.0616 |
| Y*_da / L'_da | s | 94 | -0.00966 | -0.00129 | -8.74e-05 | 0 | 0.000319 |

### Longitudinal (tabulated TF factors; phugoid stats only over rows with a complex phugoid pair)
| quantity | unit | n | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|---|
| omega_p | rad/s | 80 | 0.00788 | 0.0166 | 0.0637 | 0.132 | 0.191 |
| zeta_p | - | 80 | -0.00782 | 0.0292 | 0.105 | 0.604 | 0.767 |
| 1/T_theta1 | 1/s | 94 | -0.0027 | 0.00321 | 0.0148 | 0.0594 | 0.104 |
| 1/T_theta2 | 1/s | 94 | 0.0661 | 0.232 | 0.519 | 1.38 | 3.47 |
| omega_sp | rad/s | 94 | 0.674 | 1.06 | 2.13 | 4.81 | 10.3 |
| zeta_sp | - | 94 | 0.0316 | 0.0703 | 0.353 | 0.627 | 0.887 |

### Longitudinal force/moment derivatives (computed from tabulated body-axis values)
| quantity | unit | n | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|---|
| X_alpha (stab) = V*X_w(stab) | ft/s^2 | 94 | -77.7 | -39 | -0.00179 | 19.1 | 30.9 |
| X_alpha - g cos(gamma0) (aero part) | ft/s^2 | 94 | -110 | -71.2 | -32.2 | -13 | -1.29 |
| X_u(stab)*V/g | - | 94 | -3.01 | -1.13 | -0.33 | -0.123 | -0.099 |
| X_u(stab)*V/g, subsonic rows M<=0.8 only | - | 59 | -1.71 | -0.719 | -0.208 | -0.117 | -0.099 |
| implied L/D=-2/(X_u V/g), M<=0.8 only | - | 59 | 1.17 | 2.78 | 9.63 | 17.1 | 20.2 |
| X_u* (body, as tabulated) | 1/s | 94 | -0.0793 | -0.0411 | -0.0103 | -0.00221 | 0.00477 |
| M_u* (body, as tabulated) | 1/(ft s) | 94 | -0.00425 | -0.000592 | 0.00026 | 0.00282 | 0.0061 |
| M_u (stab) | 1/(ft s) | 94 | -0.0048 | -0.000989 | -3.95e-05 | 0.000921 | 0.00441 |
| V*M_u (stab) | 1/s^2 | 94 | -4.19 | -0.876 | -0.0229 | 1.17 | 4.35 |
| M_u(stab)/M_w(stab) | - | 94 | -0.714 | -0.0732 | 0.00865 | 0.135 | 0.66 |
| M_alpha = U0*M_w | 1/s^2 | 94 | -100 | -22.9 | -3.96 | -0.829 | -0.223 |
| M_q | 1/s | 94 | -2.8 | -1.32 | -0.528 | -0.195 | -0.0482 |
| Z_w | 1/s | 94 | -3.59 | -1.45 | -0.579 | -0.291 | -0.111 |

### Inertia ratios (body axes as tabulated). Upper block: all 94 rows; lower block: 18 distinct aircraft/inertia sets
| quantity | unit | n | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|---|
| Iy/Ix (rows) | - | 94 | 0.887 | 1.14 | 4.74 | 20.1 | 21.9 |
| Iz/Ix (rows) | - | 94 | 1.84 | 2.02 | 5.47 | 20.6 | 22.5 |
| Ixz/Ix (rows) | - | 94 | -0.489 | -0.29 | 0.0871 | 0.602 | 0.742 |
| (Iz-Ix)/Iy (rows) | - | 94 | 0.891 | 0.918 | 0.95 | 1.03 | 2.03 |
| (Iy-Iz)/Ix (rows) | - | 94 | -6.72 | -1.05 | -0.878 | -0.341 | -0.242 |
| (Ix-Iy)/Iz (rows) | - | 94 | -0.931 | -0.929 | -0.371 | -0.0702 | 0.0616 |
| Iy/Ix (distinct sets) | - | 18 | 0.887 | 1.14 | 2.63 | 15.7 | 21.9 |
| Iz/Ix (distinct sets) | - | 18 | 1.84 | 2.04 | 3.49 | 15.9 | 22.5 |
| Ixz/Ix (distinct sets) | - | 18 | -0.489 | -0.124 | 0.0639 | 0.427 | 0.742 |
| (Iz-Ix)/Iy (distinct sets) | - | 18 | 0.891 | 0.911 | 0.95 | 1.01 | 2.03 |
| (Iy-Iz)/Ix (distinct sets) | - | 18 | -6.72 | -1.01 | -0.853 | -0.459 | -0.242 |
| (Ix-Iy)/Iz (distinct sets) | - | 18 | -0.931 | -0.922 | -0.385 | -0.0709 | 0.0616 |

Distinct sets with the XB-70A nominal Iy replaced by the Fig. XI-1 value 19.9e6 (table value 1.00e7 violates Iz <= Ix+Iy):
| quantity | unit | n | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|---|
| Iy/Ix (distinct, XB-70A Iy corrected) | - | 18 | 0.887 | 1.14 | 2.63 | 15.7 | 21.9 |
| Iz/Ix (distinct, XB-70A Iy corrected) | - | 18 | 1.84 | 2.04 | 3.49 | 15.9 | 22.5 |
| Ixz/Ix (distinct, XB-70A Iy corrected) | - | 18 | -0.489 | -0.124 | 0.0639 | 0.427 | 0.742 |
| (Iz-Ix)/Iy (distinct, XB-70A Iy corrected) | - | 18 | 0.891 | 0.911 | 0.95 | 1 | 1.03 |
| (Iy-Iz)/Ix (distinct, XB-70A Iy corrected) | - | 18 | -1.22 | -1.01 | -0.853 | -0.459 | -0.242 |
| (Ix-Iy)/Iz (distinct, XB-70A Iy corrected) | - | 18 | -0.931 | -0.922 | -0.461 | -0.0709 | 0.0616 |

