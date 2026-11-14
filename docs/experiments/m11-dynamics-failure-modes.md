# M11 study: failure modes of ring-polymer dynamics methods, tier 1 (Morse OH, exact reference)

Status: **plan approved (owner, 2026-11-14); pilot in progress.** Experiment flow: this plan is
committed before any measurement; the log is appended as runs happen; results and writeup cite
the log. No engine change: the engine is used as merged (main at b95f3d1).

The whole study is tiered; this document plans **tier 1 only**. Tiers 2 (gas-phase H₂O, MACE) and 3
(liquid water, MACE, socket + cueq) get their own plans once tier 1 shows which failures are worth
chasing and at what P, Δt and run length (budgets in docs/experiments/m10-profiling-spike.md, Q5).

## Plan

### Why tier 1

The merged methods (M7: RPMD, TRPMD; M7b: PA-CMD; P = 1 classical MD as control) approximate the
Kubo-transformed correlation functions of the centroid. Their known failures are in vibrational
spectra of stiff bonds:
- **RPMD: spurious resonances.** Physical vibrations mix with internal ring-polymer modes, giving
  extra peaks or split peaks (Rossi, Ceriotti, Manolopoulos, J. Chem. Phys. 140, 234116 (2014),
  arXiv:1406.1074; their earlier references Habershon 2008, Shiga 2008).
- **CMD / PA-CMD: the curvature problem.** At low temperature stretch bands red-shift and
  broaden. The secondary sources found on 2026-11-14 attribute it to the ring polymer spreading
  along angular coordinates (Witt et al. 2009; Ivanov et al. 2010; reviewed in arXiv:2508.16246).
  So it should need more than one dimension: a 1D oscillator should not show it.
- **TRPMD:** damps the spurious resonances at the price of broader peaks (same 2014 paper).
- **Classical MD:** no zero-point energy; the peak sits near the harmonic frequency at low T,
  above the quantum fundamental.

A Morse oscillator with OH-like parameters makes each of these visible in hours of CPU, and its
exact Kubo-transformed correlation functions follow from a numerically exact eigenstate sum. So
every failure is measured against the exact answer, not against another approximation, and can be
repeated across temperatures and dimensionality cheaply.

### Questions

- **Q1.** For each method, how far is the centroid velocity spectrum (and C_vv(t)) from exact: peak
  position, width, spurious features? At 300 K and 100 K.
- **Q2.** Is the PA-CMD red shift a multidimensional effect: present for the 3D radial oscillator,
  absent (or much smaller) for the 1D one, and growing as T falls?
- **Q3.** Do RPMD's spurious resonances appear in 1D, in 3D, or both, and does TRPMD remove them?
- **Q4.** Are the conclusions converged in P, Δt, and (PA-CMD) Ω? Which settings would tiers 2–3 need?

### Hypotheses (written before running)

- **H1.** Classical (P = 1): fundamental peak blue of exact by roughly 2χ (≈ 170 cm⁻¹ for these
  parameters) at 100 K, less at 300 K; width small in 1D.
- **H2.** PA-CMD: 1D peak within 30 cm⁻¹ of exact at both T; 3D peak red of exact, by more at
  100 K than at 300 K.
- **H3.** RPMD: in at least one system/T, spectral weight appears where the exact spectrum has
  none (beyond the noise level defined below), at or near internal-mode frequencies; TRPMD
  removes it and its main peak is broader than RPMD's.
- **H4.** All quantum methods give the exact static quantities (⟨x²⟩-type values, C(0)) within
  error once P is converged, since they sample the same quantum Boltzmann distribution. A failure
  here is a bug or unconverged P, not a method failure.

These are expectations, not claims; the study reports whatever is measured.

### Systems

One particle type with the OH reduced mass μ = m_O·m_H/(m_O + m_H) (amu values as the engine
configs use: 15.999, 1.008 → μ = 0.948 336 amu). Morse V(r) = D (1 − e^{−a(r − r_e)})² with
D = 0.18748 Eh, a = 1.1605 bohr⁻¹, r_e = 1.8324 bohr (OH-like; ω_e = a√(2D/μ) ≈ 3750 cm⁻¹,
anharmonicity χ = ω_e²/(4D) ≈ 86 cm⁻¹, so the 0→1 transition ≈ 3580 cm⁻¹; exact values come from
the reference script). Gas phase (no cell), external field (no particle interactions).

- **S1 (1D):** V = Σ_i Σ_{d∈x,y,z} V(q_{i,d}), so every Cartesian component of every particle is an
  independent 1D Morse oscillator (as the M7 double-well rung uses `DoubleWell`). The Y-4
  estimator then gives 3× the 1D correlation function; C_xx contains the constant ⟨x⟩², which the
  reference includes.
- **S2 (3D radial):** V = Σ_i V(|q_i|): a particle bound to a fixed centre, i.e. the relative
  coordinate of a rotating, vibrating OH diatomic. Stretch couples to rotation, which is what the
  curvature problem needs.

N = 64 non-interacting particles per run (more samples per segment at no extra request count).

### Exact reference (`studies/m11-dynamics-failure/exact.py`, numpy, `pixi run -e ref`)

Kubo-transformed C̃_AB(t) = (1/(βZ)) Σ_{n,m} (e^{−βE_n} − e^{−βE_m})/(E_m − E_n) A_nm B_mn e^{−i(E_m−E_n)t},
with velocity matrix elements v_nm = i ω_nm x_nm.
- S1: sinc DVR (as `gen_ipi_doublewell.py`), grid converged until E and x_nm of every state with
  Boltzmann weight > 1e-12 are stable to 1e-10; result × 3 to match Y-4.
- S2: radial DVR for each l (centrifugal term l(l+1)/(2μr²)), l up to where the weight is < 1e-12;
  transitions l → l ± 1 with the angular factor (summed over M and Cartesian components the
  weight of a pair is max(l, l')·|∫u_{nl} r u_{n'l'} dr|²).
- Self-checks: ⟨V⟩ and ⟨r²⟩ from the eigenstates against an independent quadrature at each T; the
  harmonic limit (a → 0 at fixed ω_e) against the analytic Kubo C_xx; C̃(0) against the static
  ⟨x²⟩ the PIMD runs measure (H4).

### Force client (`studies/m11-dynamics-failure/morse_client.py`)

i-PI client for S1/S2, importing the locked M5 `ipi_client.py` (`connect`, `serve`) by path, as the
M6/M8/M10 clients do; U-2 constants not needed (everything stays in atomic units, K-2). Gas phase
only (h must be zero). Stateless. Energies and forces checked against finite differences (1e-7
relative) before any run.

### Protocol (all runs: `[dynamics]` segments, Y-1)

- Temperatures 300 K and 100 K. βħω_e ≈ 18 at 300 K and 54 at 100 K: the oscillator is in its
  ground state in both; 100 K is where the curvature problem should be strongest.
- P: chosen in the pilot so that ⟨V⟩ (primitive and virial, E-1/E-2) is within 1 % of exact; the
  expectation is P = 32 at 300 K and P = 96 at 100 K. P = 1 for the classical control.
- Δt = 0.1 fs to start (stretch period 9 fs); the pilot halves it once per system and checks C_vv.
- Methods: classical (P = 1, `rpmd`), `rpmd`, `trpmd` (λ = 0.5, the M7 default), `pa-cmd` with
  Ω = 16 000 cm⁻¹ (≈ 4× the stretch; Ω·Δt = 0.3 at 0.1 fs) and a second run at 32 000 cm⁻¹ at
  Δt/2 for adiabaticity (the M7b brief's check). `cmd_lambda` = 1.0 (as the M7b validation; it
  is a required key).
- Segments: `decorrelation_steps` = 1000 (100 fs, with τ0 = 20 fs PILE-L), `segment_steps` =
  20 000 (2 ps; spectral resolution ≈ 17 cm⁻¹), `corr_stride` = 2, `n_lags` = 10 001.
  100 segments per run as the starting point; the pilot sets the final count from the σ achieved.
- Matrix: {S1, S2} × {300 K, 100 K} × {classical, RPMD, TRPMD, PA-CMD} = 16 production runs, plus
  the convergence runs of Q4 (P, Δt, Ω), roughly 6 more.
- Seeds fixed per run and logged; every config committed under `studies/m11-dynamics-failure/configs/`.

### Analysis (`studies/m11-dynamics-failure/analyse.py`)

- Spectrum: I(ω) ∝ ∫ C_vv(t) w(t) cos ωt dt with a Hann window over the 2 ps lag range, applied
  identically to the exact C_vv sampled on the same lag grid, so window broadening cancels in the
  comparison.
- Per run: peak position (parabolic fit to the top three bins), FWHM, spurious weight (integrated
  |I_method − I_exact| outside ±150 cm⁻¹ of the exact peak, normalised by the total), and the
  time-domain error ‖C_vv − C_exact‖ over 0–500 fs relative to C_exact(0).
- Errors: segment-blocked σ as Y-4 defines, propagated to the spectral metrics by resampling
  segments (200 bootstrap draws, fixed seed).

### What counts as a finding

- A **failure** is a deviation from exact larger than 5σ (statistical) plus the larger of the Δt
  and P sensitivities measured in Q4. Smaller deviations are reported as "not resolved".
- **Repeated** means seen in both temperatures or both systems with the same sign; the user's
  goal is failures that come out more than once, so single-case effects are flagged as such.
- A static quantity outside 5σ of exact (H4) stops the run series: it is a bug or an unconverged P
  and gets a log entry and an owner decision before continuing.

### Budget and resources

- **Machine:** CPU only; the GPU is not used (a tier 2/3 job could run beside this).
- **Cost model:** one request per bead per step (K-4), so the time per step ≈ P × the client round
  trip. The pilot measures the round trip; at an assumed 50 µs, P = 32 is 1.6 ms per step and a
  100-segment run (2.1 M steps) ≈ 1 h; P = 96 ≈ 3 h. With K = 2–4 clients per run and 2–3 runs in
  parallel (RAM is not an issue at this size), the 22 runs fit in about 1–2 days of wall time.
- **Cap:** 72 h of wall time for tier 1; stop and report if the pilot projects more.
- **Storage:** `observables.csv`, `correlations.csv`, `segments.csv` only (`trajectory_every`
  large); outputs under `studies/m11-dynamics-failure/out/` (gitignored), results tables in
  `results/` committed.

### Pilot (first measurement step)

1. Client checks (finite differences) and exact reference self-checks.
2. Round-trip time per request at N = 64 (K = 1, 2, 4), to fix the budget.
3. P and Δt choice per T (static ⟨V⟩ vs exact; short C_vv at Δt and Δt/2).
4. One short run per method (10 segments) to check outputs, σ per segment, and the run count needed.

The pilot's numbers go into the log, and the production matrix is adjusted only by the rules
above (P, Δt, segment count); any other change is logged with the reason before production.

### What would change the plan

- Round trip ≫ 50 µs (cost > cap): reduce N or segments, or propose a small in-process `Morse`
  potential as a milestone (owner decision; not done in this study).
- The exact reference fails its self-checks: fix it before any comparison.
- PA-CMD not adiabatic at 16 000 cm⁻¹ (results differ from 32 000 cm⁻¹ beyond σ): use the larger
  Ω for production at the Δt it needs.

## Log

### 2026-11-14 01:00 – 01:50 Pilot steps 1–3 (engine at 26761ec; CPU only, GPU idle)

Scripts: `morse_client.py`, `exact.py`, `run.py`, `statics.sh`, `statics.py` (studies/m11-dynamics-failure/).
Outputs: `out/pilot/{timing,statics}/`.

- **Client check** (`morse_client.py --check`): max |F_fd − F|/max |F| = 2.2e-10 (S1), 8.6e-11 (S2).
- **Exact reference** (`exact.py --check`): μ = 1728.565 181 m_e = 0.948 256 amu (the plan's
  0.948 336 was a typing slip; the code uses the masses as written). ω_e = 3751.28 cm⁻¹.
  Grid 600 vs 900 points: C_xx and C_vv agree to ≤ 7e-12 relative (both systems, 300 and 100 K).
  Sum rule C̃_vv(0) = 3/(βμ): satisfied to ≤ 8e-12. Harmonic limit vs analytic Kubo C_xx: 8e-13.
  The first version failed the sum rule (−50 % at 100 K): pairs whose final state is not a kept
  initial state were counted once instead of twice; fixed (`mirror()`), then all checks pass.
  S2 needs l ≤ 18 (300 K), l ≤ 10 (100 K).
- **Clarification of H4 / the reference self-checks:** the static quantity the runs measure at lag
  0 is the Kubo-transformed C̃_xx(0) (centroid second moment), not ⟨x²⟩; C̃_vv(0) = 3/(βμ)
  exactly. The comparisons use those.
- **Round trip** (`out/pilot/timing/`, Δwall between 200 and 1200 plain PIMD steps): S1, P = 32:
  4.06 / 2.61 / 1.86 ms per step with K = 1 / 2 / 4 (58 µs per request at K = 4). S2, P = 192:
  14.5 / 11.1 / 8.1 ms per step with K = 4 / 8 / 12.
- **Statics vs exact** (`statics.py`; 2000 + 20 000 PILE-L steps, Δt = 0.1 fs, N = 64; % of exact,
  σ from 10 blocks; E_vir shown, ⟨V⟩ and E_prim agree with it within ≈ 2σ):

  | | P = 16 | 32 | 64 | 48 | 96 | 128 |
  |---|---|---|---|---|---|---|
  | S1 300 K | −12.8 ± 0.04 | −3.95 ± 0.07 | −1.04 ± 0.07 | | | |
  | S1 100 K | | | | −12.8 ± 0.05 | −3.77 ± 0.06 | −2.17 ± 0.04 |
  | S2 300 K | −11.3 ± 0.07 | −3.62 ± 0.16 | −1.13 ± 0.11 | | | |
  | S2 100 K | | | | −12.4 ± 0.09 | −3.61 ± 0.06 | −2.09 ± 0.07 |

  The error falls as 1/P² and depends on P/T only (100 K at P = 96 equals 300 K at P = 32).
  Δt = 0.05 fs gives the same values within σ (S1/S2 at 300 K P = 32 and 100 K P = 96).
- **P choice (plan rule: within 1 % of exact):** P = 64 at 300 K (≈ −1.0 %), P = 192 at 100 K
  (≈ −0.9 % by the 1/P² trend from 96 and 128), twice the plan's expectation. Δt = 0.1 fs kept.
- **Budget re-estimate:** throughput is CPU-bound at ≈ 24 000 bead-steps per second on all 12
  cores (P = 192, K = 12). 100 segments for the 6 quantum runs per temperature plus ≈ 6
  convergence runs ≈ 4.2e9 bead-steps ≈ 49 h: inside the 72 h cap; the segment count from step 4
  may lower it.
- **Change before production (logged per the plan):** `correlations.csv` carries the mean and σ of
  C_vv per lag but no per-segment functions, so the spectral metrics' σ cannot be bootstrapped
  over segments. Instead each production run is split into independent replicas (different seeds,
  equal segment counts), which also run in parallel; the metrics' σ is the standard error over
  replicas (`analyse.py`). The time-domain σ uses the engine's per-lag σ.

### 2026-11-14 01:55 Pilot step 4 started (`pilot_dyn.sh`; `out/pilot/dyn/`)

300 K, 10 segments per run, P = 64 (classical P = 1), K = 3, three runs at a time, both systems.

### 2026-11-14 02:25 Pilot step 4 results (`analyse.py out/pilot/dyn/*`; 10 segments, one replica each)

Wall ≈ 1 400 s per P = 64 run (three at a time, K = 3), ≈ 200 s classical.

| | centre (cm⁻¹) | width (cm⁻¹) | C̃_xx(0) vs exact |
|---|---|---|---|
| S1 exact | 3580.3 | 8.0 (window) | |
| S1 classical | 3733.5 | 14.2 | −2.85 % (classical, not H4) |
| S1 RPMD | 3649.7 | 36.6 | −0.03 % |
| S1 TRPMD | 3644.6 | 123.5 | −0.03 % |
| S1 PA-CMD | 3623.5 | 73.5 | −0.04 % |
| S2 exact | 3595.5 | 120.5 (P/R branches) | |
| S2 classical | 3717.6 | 125.1 | +0.09 % |
| S2 RPMD | 3634.6 | 139.3 | −0.06 % |
| S2 TRPMD | 3590.7 | 217.7 | −0.03 % |
| S2 PA-CMD | 3446.9 | 188.6 | −0.03 % |

- H4 holds in the pilot (all quantum C̃_xx(0) within 0.06 %). C_vv deviations are 10–900 engine σ,
  dominated by the frequency shifts (a 70 cm⁻¹ shift dephases fully within 500 fs).
- No σ for the spectral metrics yet (one replica).
- **Metric changes before production** (logged per the plan; decided on the pilot's form of the
  spectra, not on which method they favour):
  1. *Peak* is kept but is not a band position for S2: the exact S2 spectrum has P and R branches
     and no Q branch, so its maximum is the R-branch top (3683.7 cm⁻¹). The **band centre** (first
     moment over 2800–4400 cm⁻¹) is the primary position metric for both systems.
  2. *Spurious weight*: the plan's ∫|I − I_exact| outside ±150 cm⁻¹ of the exact peak measured
     misplacement (any shift scores ≈ 1 for S1) and, for S2, the rotational band. Replaced by the
     fraction of ∫|I| above 1500 cm⁻¹ lying more than 400 cm⁻¹ from the spectrum's own band
     centre (exact: 0.000 for both systems; pilot: 0.000–0.13).
- **Segment count:** the effects seen are 40–150 cm⁻¹. Production uses 4 replicas × 10 segments per
  (system, T, method), the pilot runs as replica 0 at 300 K (same settings, own seeds); more
  replicas are added only if a metric's σ is too large to classify an effect (plan rule).
- **Q4 runs** (2 replicas each, 300 K): P = 128 (RPMD, PA-CMD), Δt = 0.05 fs (RPMD), PA-CMD at
  Ω = 32 000 cm⁻¹ with Δt = 0.05 fs; both systems. 100 K convergence is inferred from P/T scaling
  of the statics unless the 300 K checks fail.
- **Budget:** 72 runs (`production.sh`), ≈ 3 h for 300 K, ≈ 3 h for Q4, ≈ 9–10 h for 100 K.

### 2026-11-14 02:30 Production started (`production.sh`; `out/prod/`, `out/production.log`)
