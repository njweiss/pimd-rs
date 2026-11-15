# M11 study: failure modes of ring-polymer dynamics methods, tier 1 (Morse OH, exact reference)

Status: **tier 1 complete (2026-11-15); results and writeup below; Ω rule: owner accepted option (a).** Experiment flow: this plan is
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

### 2026-11-14 05:05 300 K replica set complete (24 production runs, all rc 0)

- Wall per run: 1352–1471 s (quantum, P = 64, K = 3); 206–209 s (classical). Production at 300 K
  took 2 h 25 min, inside the ≈ 3 h budget. Q4 runs started at 04:52.
- `analyse.py out/pilot/dyn/* out/prod/*_T300_*_r[123]` (n = 4 replicas per group; σ = standard
  error over replicas). Band centre / width / spurious weight (exact in brackets):

  | | S1 centre | S1 width | S1 spur. | S2 centre | S2 width | S2 spur. |
  |---|---|---|---|---|---|---|
  | exact | 3580.3 | 8.0 | 0.000 | 3595.5 | 120.5 | 0.000 |
  | classical | 3733.4 ± 0.1 | 14.4 ± 0.3 | 0.000 | 3717.4 ± 0.3 | 126.9 ± 1.3 | 0.000 |
  | RPMD | 3650.4 ± 0.3 | 36.7 ± 0.5 | 0.002 | 3635.1 ± 0.5 | 140.1 ± 1.3 | 0.101 |
  | TRPMD | 3644.6 ± 0.1 | 123.1 ± 0.2 | 0.041 | 3590.6 ± 0.2 | 217.7 ± 0.3 | 0.126 |
  | PA-CMD | 3623.4 ± 0.1 | 73.2 ± 0.2 | 0.013 | 3446.5 ± 0.2 | 188.2 ± 0.7 | 0.063 |

- Replica σ of the band centre is ≤ 0.5 cm⁻¹ for every group, far below every method–exact
  difference (≥ 4.9 cm⁻¹, TRPMD S2): no extra replicas needed at 300 K. The S2 peak position is
  noisy (σ up to 55 cm⁻¹, two-branch band); the centre stays the primary metric, as planned.
- Quantum C̃_xx(0) within 0.05 % of exact in all six quantum groups (H4).
- Every pilot number reproduces within 1 cm⁻¹ (centre) at n = 4.

### 2026-11-14 08:00 – 09:15 100 K PA-CMD unstable at Δt = 0.1 fs; Δt changed (plan rule: Δt)

- Q4 runs at 300 K all finished rc 0 (16 runs; P = 128 runs ≈ 2 900 s). The 100 K series began at
  07:55. `s1_T100_pa-cmd_r0` failed within 0.5 s: `backend: bead 0: client 0: non-finite energy
  inf`. Reproduced in scratch short runs (same config, 50 + 100 + 200 steps):
  - deterministic (two seeds); fails at 100 K for P = 64, 128, 160, 192, and at 150 K; passes at
    200, 250, 300 K; fails at Ω = 12 000 and 32 000 cm⁻¹ too; RPMD at the same settings passes.
  - an instrumented client shows every bead at r_e at step 0, a spread of 0.1 bohr after step 1,
    2.4 bohr by step 3 and 10¹⁷ bohr at step 4: Verlet blowing up, not a slow drift.
- Cause (not an engine fault; Y-6 as specified): with dynamical masses m·s_k, s_k = (ω_k/Ω)², the
  physical curvature μω_e² acting on mode k oscillates at ω_e/√s_k = ω_e·Ω/ω_k, integrated by the
  velocity-Verlet force step. For the slowest internal mode ω_1 = 2ω_P sin(π/P) ≈ 2πk_BT/ħ, so this
  frequency grows as 1/T: ω_e·Ω/ω_1 Δt = 0.86 at 300 K (P = 64), 1.29 at 200 K, 1.73 at 150 K and
  2.59 at 100 K (P = 192), past the harmonic Verlet limit of 2. The repulsive wall is stiffer than
  the harmonic curvature, which is why 150 K already fails.
- S2 did not crash at 100 K, Δt = 0.1 fs (5 000-step scratch run), but is silently wrong: ⟨V⟩ =
  0.103 Eh per particle against exact 0.0043 (beads heated out of the well). A crash is the better
  of the two outcomes; this is the more dangerous one and is recorded as a PA-CMD failure mode
  (practical, not of the method's dynamics): the stable Δt at fixed Ω scales with T.
- At 100 K, P = 192, S1: Δt = 0.05 fs gives ⟨V⟩ = 0.01283 Eh and Δt = 0.0333 fs 0.01281 Eh against
  exact 0.012819 (short runs).
- Change: 100 K PA-CMD production uses Δt = 0.1/3 fs, the same ω_e·Ω/ω_1·Δt (0.86) as the 300 K
  production, with decorrelation 3 000, segment 60 000, corr_stride 6 and n_equil 6 000 steps, so
  the simulated times and the lag grid (0.2 fs) are unchanged; 10 segments, replicas r0–r3, tags
  `*_T100_pa-cmd_dt0.033_r*`. A Δt check at 0.05 fs (same simulated times; r0, r1, tags
  `*_T100_pa-cmd_dt0.05_r*`) is added to Q4. Cost ≈ 3× (2×) a 100 K RPMD run each; projected end
  of production ≈ 2026-11-15 06:00, inside the 72 h cap.
- The failed directory is moved to `out/failed/`. The queue was stopped (xargs killed) with three
  100 K runs in flight (s1 trpmd r0, s2 rpmd r0, s2 trpmd r0), which finish and are kept;
  `production.sh` is restarted with the new jobs first (longest first) and skips runs in flight.

### 2026-11-14 17:15 First 100 K results (PA-CMD at Δt = 0.033 fs, r0–r2; other methods r0 only)

- All six PA-CMD runs rc 0 at ≈ 13 400–13 900 s. C̃_xx(0) within 0.03 % of exact (statics fine).
- `analyse.py` crashed on S2 PA-CMD: the spectrum's maximum inside 2800–4400 cm⁻¹ sat on the
  window's lower edge, so the three-bin refinement had no neighbour. Fixed (an edge maximum is
  reported unrefined, as the edge value); no other change to the metrics.
- S2 PA-CMD at 100 K: the stretch band has left the analysis window. Over ω ≥ 1500 cm⁻¹ the
  maximum is at 2124 cm⁻¹ and the first moment at 2340 cm⁻¹ (exact: R-branch peak 3650.6, centre
  3609.5); 65 % of the weight lies in 2000–2800 cm⁻¹. At 300 K the same method's band was at
  3359 / 3452 cm⁻¹. This is the size of red shift reported for CMD's curvature problem in
  radial/rotating systems at low T. The planned window metrics (centre 3423.7, spurious 0.951)
  describe the truncated tail, not the band.
- Analysis addition, made after seeing this result and recorded as such: the final tables also
  give, for every group, the maximum and first moment over ω ≥ 1500 cm⁻¹ (above the rotational
  band). The planned window metrics stay as defined and are reported unchanged beside them.
- S1 PA-CMD at 100 K: centre 3617.3 ± 0.0, width 73.5, much as at 300 K (3623.4, 73.2). RPMD and
  TRPMD r0 at 100 K are also close to their 300 K values on both systems.

### 2026-11-15 05:40 Production finished; analysis fixes (09:30)

- `production.sh` ended at ≈ 05:40 (PID 3729061): 76 run directories, all rc 0 with
  `correlations.csv`, apart from the moved `out/failed/s1_T100_pa-cmd_r0`. Wall per run: 100 K
  RPMD/TRPMD 3 995–4 627 s, 100 K PA-CMD at Δt = 0.033 fs 13 159–13 931 s. Production wall time
  02:30 on 2026-11-14 to 05:40 on 2026-11-15 ≈ 27 h, inside the 72 h cap.
- Every run's `config.toml` is now copied to `configs/<out subdir>_<tag>.toml` (85 files; the plan
  required it and it had not been done).
- **Q4 oversight in `production.sh`:** the 300 K Δt = 0.05 fs runs (RPMD, and PA-CMD at
  Ω = 32 000 cm⁻¹) kept `corr_stride` = 2, so their lag range is 1 ps, not 2 ps. Their spectra use
  a narrower Hann window (exact S1 width 15.4 instead of 8.0 cm⁻¹). Their comparisons are
  therefore made with every run's C_vv truncated at 1 ps (`analyse.py --tmax-fs 1000`,
  `results/metrics_1ps.*`); band centres move ≤ 0.4 cm⁻¹ under the truncation.
- **`analyse.py` bug:** the exact reference was cached by (system, T, number of lags), and the
  1 ps runs have the same number of lags on a different grid, so they were compared with the 2 ps
  exact. The key now includes the lag spacing. Only those runs' "exact" columns and time-domain
  errors changed.
- `results/`: `metrics.{txt,json}` (2 ps, every group), `metrics_1ps.{txt,json}`, and
  `spectra.png` (`plot_spectra.py`: production spectra of all methods with exact, and the internal
  mode frequencies).
- **Ω rule triggered:** at 300 K, PA-CMD at Ω = 32 000 cm⁻¹ (Δt = 0.05 fs) moves the S2 centre by
  −11.0 cm⁻¹ (σ 0.4) and the widths from 73 to 54 (S1) and 188 to 159 cm⁻¹ (S2). The plan says
  production then uses the larger Ω. It was not rerun. At 100 K the stable Δt at 32 000 cm⁻¹ would
  be 1/60 fs, about 20 h of wall time for the eight runs. Here Ω sensitivity is counted in the
  PA-CMD threshold instead (as Δt and P are), and the PA-CMD widths are reported as Ω-dependent.
  Owner decision pending (see Results).

## Results

Sources: `studies/m11-dynamics-failure/results/metrics.{txt,json}` (2 ps), `metrics_1ps.*` (Q4 Δt
comparisons), `spectra.png`; log entries 2026-11-14 05:05, 17:15 and 2026-11-15 09:30. n = 4
replicas per production group (10 segments each; the pilot runs are 300 K replica 0); σ is the
standard error over replicas. "Centre" is the first moment over 2800–4400 cm⁻¹ (planned);
"full-range centre" is over ω ≥ 1500 cm⁻¹ (added after the first 100 K results, log 17:15).

### Production (P = 64 at 300 K, 192 at 100 K; Δt = 0.1 fs, PA-CMD at 100 K 0.033 fs; Ω = 16 000 cm⁻¹)

Exact centres: S1 3580.3 (both T); S2 3595.5 (300 K), 3609.5 (100 K). Exact widths are in brackets.

| System, T | Method | Centre − exact | Full-range centre − exact | Width (exact) | Spurious | C̃_xx(0) |
|---|---|---|---|---|---|---|
| S1, 300 K | classical | +153.2 ± 0.1 | +153.2 ± 0.1 | 14.4 (8.0) | 0.000 | −2.85 % |
| S1, 300 K | RPMD | +70.1 ± 0.3 | +70.4 ± 0.3 | 36.7 (8.0) | 0.002 | −0.04 % |
| S1, 300 K | TRPMD | +64.3 ± 0.1 | +71.8 ± 0.1 | 123.1 (8.0) | 0.041 | −0.03 % |
| S1, 300 K | PA-CMD | +43.1 ± 0.1 | +46.2 ± 0.1 | 73.2 (8.0) | 0.013 | −0.04 % |
| S1, 100 K | classical | +165.6 ± 0.1 | +165.6 ± 0.1 | 8.1 (8.0) | 0.000 | −3.08 % |
| S1, 100 K | RPMD | +69.9 ± 0.5 | +70.1 ± 0.5 | 37.6 (8.0) | 0.001 | −0.04 % |
| S1, 100 K | TRPMD | +64.3 ± 0.1 | +72.5 ± 0.1 | 125.6 (8.0) | 0.043 | −0.03 % |
| S1, 100 K | PA-CMD | +37.1 ± 0.1 | +40.2 ± 0.0 | 73.5 (8.0) | 0.014 | −0.03 % |
| S2, 300 K | classical | +121.9 ± 0.3 | +121.9 ± 0.3 | 126.9 (120.5) | 0.000 | +0.07 % |
| S2, 300 K | RPMD | +39.6 ± 0.5 | −46.3 ± 0.7 | 140.1 (120.5) | 0.101 | −0.05 % |
| S2, 300 K | TRPMD | −4.9 ± 0.2 | −35.8 ± 0.1 | 217.7 (120.5) | 0.126 | −0.03 % |
| S2, 300 K | PA-CMD | −149.0 ± 0.2 | −143.7 ± 0.2 | 188.2 (120.5) | 0.063 | −0.03 % |
| S2, 100 K | classical | +131.3 ± 0.2 | +131.3 ± 0.2 | 73.6 (67.6) | 0.000 | +6.02 % |
| S2, 100 K | RPMD | +12.2 ± 1.3 | −64.7 ± 2.1 | 187.5 (67.6) | 0.125 | −0.02 % |
| S2, 100 K | TRPMD | −14.6 ± 0.5 | −87.7 ± 0.5 | 226.3 (67.6) | 0.179 | −0.02 % |
| S2, 100 K | PA-CMD | −184.9 ± 1.4 | −1271.2 ± 1.8 | 462.3 (67.6) | 0.951 | −0.03 % |

The S2 100 K PA-CMD band lies outside the planned window (maximum at 2126 ± 5 cm⁻¹ against exact
3651); its window centre and spurious weight describe the truncated tail, and the full-range
column is the meaningful one. Classical C̃_xx(0) is not a Kubo quantity and is not expected to
match (H4 concerns the quantum methods).

### Q4 sensitivities (change in centre, cm⁻¹; σ ≤ 1.3)

| Change | RPMD S1 | RPMD S2 | PA-CMD S1 | PA-CMD S2 |
|---|---|---|---|---|
| P 64 → 128 (300 K, 2 ps) | +1.2 | +0.1 | +0.4 | −0.3 |
| Δt 0.1 → 0.05 fs (300 K, 1 ps) | −1.3 | −3.5 | n/a | n/a |
| Ω 16 000 → 32 000 cm⁻¹ with Δt 0.05 fs (300 K, 1 ps) | n/a | n/a | −1.9 (width 74 → 54) | −11.0 (width 189 → 159) |
| Δt 0.033 → 0.05 fs (100 K, 2 ps) | n/a | n/a | +0.4 | −1.4 (full-range +1.2) |

TRPMD had no Q4 runs; RPMD's sensitivities are used for it. Thresholds for "failure" (5σ plus the
largest sensitivity): RPMD and TRPMD ≈ 3 cm⁻¹ (S1), ≈ 6 cm⁻¹ (S2); PA-CMD ≈ 3 cm⁻¹ (S1), ≈ 12 cm⁻¹
(S2). Every centre deviation in the production table exceeds its threshold, except TRPMD S2 at
300 K (−4.9, below its ≈ 6 cm⁻¹ threshold: not resolved).

### Spectral features (`spectra.png`)

- S1: every method gives one narrow line; no spurious feature in any method at either T
  (spurious ≤ 0.002 for RPMD; TRPMD's 0.04 and PA-CMD's 0.013 are their broadened line tails).
- S2, RPMD: a separate band at 2400–2750 cm⁻¹ at 300 K (≈ 8 % of the peak height), close to the
  second internal mode ω_2 = 2616 cm⁻¹; at 100 K smaller features near 1700 cm⁻¹ (ω_4 = 1746),
  2550 and 4080 cm⁻¹. The exact spectrum has no weight there. RPMD also smooths the exact P/R
  rotational structure into one envelope.
- S2, TRPMD: no separate bands; a smooth envelope with a broad low baseline (≈ 2 % of peak height
  across 1500–3000 cm⁻¹), which is what its spurious-weight number counts.
- S2, PA-CMD: a single broad band, 150 cm⁻¹ red at 300 K and centred near 2100–2300 cm⁻¹ at 100 K.

## Writeup

**Questions.**
- **Q1.** No method reproduces the exact OH stretch band at the level these runs resolve; every
  method fails on band position in at least one system (table above). Size of the worst error per
  method: classical +166 cm⁻¹ (S1, 100 K), RPMD +70 cm⁻¹ (S1), TRPMD +64 cm⁻¹ (S1) with 15× the
  exact width, PA-CMD −1271 cm⁻¹ (S2, 100 K). Among the quantum methods, PA-CMD is closest in 1D
  (+37 to +43 cm⁻¹) and furthest in 3D.
- **Q2.** Yes. The PA-CMD red shift is a multidimensional effect that grows as T falls: none in
  1D (a small blue shift, +43 → +37 cm⁻¹ from 300 to 100 K), −144 cm⁻¹ in 3D at 300 K and −1271 cm⁻¹
  at 100 K (full-range centre), with the band broadened to 6.8× the exact width. Not sensitive to
  Δt (1.4 cm⁻¹) or P (0.3 cm⁻¹); Ω sensitivity at 300 K is −11 cm⁻¹, in the direction of a larger
  shift.
- **Q3.** RPMD's spurious features appear in 3D only (both temperatures), near internal-mode
  frequencies; none in 1D at either T. TRPMD removes the separate bands, at the cost of a band 1.2–1.6×
  RPMD's width in 3D and 3.3× in 1D.
- **Q4.** Converged in P (≤ 1.2 cm⁻¹) and, for RPMD, in Δt (≤ 3.5 cm⁻¹). PA-CMD band centres are
  converged in Ω to ≈ 11 cm⁻¹ at 300 K; PA-CMD widths are not (they fall by 15–28 % from 16 000 to
  32 000 cm⁻¹). PA-CMD's stable Δt scales with T at fixed Ω (log 2026-11-14 08:00).

**Hypotheses.**
- **H1 (classical blue by ≈ 2χ ≈ 170 cm⁻¹ at 100 K, less at 300 K; narrow in 1D): supported.**
  +166 (100 K) and +153 cm⁻¹ (300 K) in S1, +131 and +122 cm⁻¹ in S2; S1 width 8.1 cm⁻¹ at 100 K
  (exact 8.0, window-limited).
- **H2 (PA-CMD 1D within 30 cm⁻¹; 3D red, more at 100 K): partly supported.** The 3D part holds
  and is the largest effect in the study. The 1D part does not: PA-CMD is +37 to +43 cm⁻¹ in 1D,
  outside the 30 cm⁻¹ expected, and not reduced at Ω = 32 000 cm⁻¹ (+41 cm⁻¹).
- **H3 (RPMD spurious weight somewhere; TRPMD removes it and broadens): supported, in 3D only.**
- **H4 (quantum statics exact): supported.** C̃_xx(0) within 0.05 % of exact in all 22 quantum
  groups (including Q4).

**Failures that repeat** (the owner's goal; "repeated" = both T or both systems, same sign):
1. Classical blue shift: both systems, both T (+122 to +166 cm⁻¹).
2. RPMD and TRPMD blue shift of the 1D line: both T (+64 to +70 cm⁻¹), Δt- and P-converged. In
   3D they are within −88 to +40 cm⁻¹ depending on the centre definition (P/R branch structure).
3. PA-CMD red shift and broadening in 3D: both T, growing roughly 9× from 300 to 100 K.
4. RPMD spurious bands near internal-mode frequencies: 3D, both T.
5. TRPMD over-broadening: every case (1.8–15× the exact width).
6. PA-CMD Δt instability below ≈ 150–200 K at Δt = 0.1 fs, Ω = 16 000 cm⁻¹: as a crash in 1D and
   as a silent wrong ensemble in 3D (⟨V⟩ 24× exact). A practical failure, not one of the dynamics.

Single-case: none of the above rests on one case.

**What this means for tiers 2 and 3.**
- The informative failures need a rotating coordinate: 1D models hide the curvature problem and
  the spurious resonances. Tier 2 (gas-phase H₂O with MACE) has rotations and two stretches plus
  the bend, so it should show both. Measure at two temperatures, since the PA-CMD shift is strongly
  T-dependent.
- PA-CMD needs Δt chosen from ω_stretch·Ω/ω_1 Δt ≲ 0.9 (ω_1 ≈ 2πk_BT/ħ for large P). At 300 K and
  Ω = 16 000 cm⁻¹, 0.1 fs suffices for an OH stretch. A guard in the engine (a warning, or an
  error when the dynamical frequency of a mode exceeds the Verlet limit for a given curvature)
  would turn the silent 3D failure into a visible one. That would be an engine change through a
  milestone, not part of this study.
- Report PA-CMD widths with an Ω check; band centres moved by ≤ 11 cm⁻¹ here.
- Window-based metrics must allow the band to move: the full-range centre caught what the planned
  window missed.
- Costs per run here: RPMD/TRPMD at P/T = 0.64 K⁻¹ with N = 64 ≈ 1 400 s at 300 K, 4 400 s at
  100 K (K = 3 socket clients, CPU); PA-CMD at 100 K 3× that.

**Open item for the owner.** The plan's Ω rule (log 2026-11-15 09:30) was triggered and not
followed: production was not rerun at Ω = 32 000 cm⁻¹. Ω sensitivity is counted in the threshold
instead. No conclusion above changes with it (the 300 K S2 shift grows from −150 to −161 cm⁻¹ on the 1 ps grid).
Options:
- (a) accept as is;
- (b) rerun 300 K PA-CMD at Ω = 32 000 cm⁻¹ (8 runs, ≈ 7 h);
- (c) (b) plus 100 K at Δt = 1/60 fs (≈ 20 h more).

**Owner decision (2026-11-15):** option (a). The results stand as reported, with Ω sensitivity
counted in the PA-CMD threshold and PA-CMD widths reported as Ω-dependent; no rerun.
