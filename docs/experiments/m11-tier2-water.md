# M11 study, tier 2: dynamics failure modes in gas-phase H₂O (MACE-OFF23 small)

Status: **plan approved (owner, 2026-11-16); pilot complete; production on hold for an owner
decision (statics rule triggered, budget over the cap; see the pilot steps 3–4 log entry).** Experiment flow: this plan is
committed before any measurement; the log is appended as runs happen; results and writeup cite
the log. No engine change: the engine is used as merged (main at 9161d43).

Tier 1 (docs/experiments/m11-dynamics-failure-modes.md, owner-accepted 2026-11-15) found, against
exact Kubo results for a Morse OH stretch:
- PA-CMD red shift and broadening in 3D only, growing about 9× from 300 to 100 K;
- RPMD spurious bands near internal-mode frequencies, in 3D only;
- TRPMD broadening;
- a classical blue shift;
- a PA-CMD time-step instability whose stable Δt scales with T at fixed Ω.

Its advice was to use a molecule that rotates, two temperatures, and a PA-CMD Δt chosen from
ω_stretch·Ω/ω_1. Tier 2 asks whether these failures appear, and at what size, for a real molecule
on a machine-learned surface: three coupled modes (bend, symmetric and antisymmetric stretch) plus
rotation.

## Plan

### Questions

- **Q1.** For each method, where are the bend and stretch bands of the centroid velocity spectrum
  relative to the surface's own vibrational band origins, and how broad are they, at two
  temperatures?
- **Q2.** Does PA-CMD red-shift the stretch (and the bend) in water, and does the shift grow at
  the lower temperature as in tier 1?
- **Q3.** Do RPMD spurious features appear near internal-mode frequencies, which now fall close to
  the bend (ω_1 ≈ 1310 cm⁻¹ at 300 K for large P)? Does TRPMD remove them?
- **Q4.** Convergence in P, Δt and Ω; the cost per method and temperature that tier 3 (liquid
  water) would face.

### Hypotheses (written before running)

- **H1.** Classical (P = 1): stretch band blue of the band origins by roughly the tier-1 amount
  (+100 to +170 cm⁻¹), the bend by much less (its anharmonicity is smaller).
- **H2.** PA-CMD: the stretch is red of the band origins at 300 K and further red at 150 K; the
  bend shifts much less.
- **H3.** RPMD: spectral weight away from the bands, near ω_k, in at least one temperature; TRPMD
  removes it and broadens the bands.
- **H4.** All quantum methods agree on the statics: the virial energy per molecule matches
  E_ZPE + 3k_BT (zero-point energy of the surface from the reference, plus classical translation
  and rotation; vibrational excitation is below 1e-3 at 300 K) within error once P is converged.

These are expectations, not claims; the study reports whatever is measured.

### System

N independent H₂O molecules (N set in the pilot; 64 to start), gas phase, engine masses
(O 15.999, H 1.008 amu). Each molecule starts at the surface's minimum geometry with a random
orientation. Molecules sit on a 15 Å grid but **do not interact at any distance**: the study's
force client builds MACE edges within a molecule only (below). This is the tier-1 "N independent
particles" design, so translation and diffusion cannot bring molecules into contact.

### Potential and force client (`studies/m11-dynamics-failure/water_client.py`)

- MACE-OFF23 small, `ff_models/mace_off23/MACE-OFF23_small.model` (gitignored; licence accepted by
  the owner with weights kept out of the repo).
- The client subclasses the supported M10a client (`crates/pimd-torch/scripts/mace_client.py`,
  K-7) and changes one thing: an edge i → j is kept only when i and j belong to the same molecule
  (atoms 3m, 3m + 1, 3m + 2). Everything else is unchanged: P-6 edges in f64, E0 in f64, U-2
  constants, gas phase only.
- Device and backend set in the pilot: CUDA with e3nn in f64 is the default. Use cueq or f32 only
  if the pilot shows that e3nn f64 does not fit the budget and that the faster route keeps NVE RPMD
  energy drift and band positions within σ.
- One run on the GPU at a time (the project rule), with K clients of that run sharing it.
- Checks before any run:
  - forces against finite differences of the client's energy;
  - one molecule from the client against the M10a client on the same geometry (identical);
  - N molecules against N × one molecule (no cross terms).

### Reference (`studies/m11-dynamics-failure/water_ref.py`)

There is no exact Kubo spectrum for a rotating water molecule on this surface within reach, so
the reference is the surface's own **J = 0 vibrational band origins**: ν2 (bend), ν1, ν3, the
zero-point energy, and 2ν2. Method:
- a 3D DVR in Radau coordinates (R1, R2, θ), whose J = 0 kinetic energy is orthogonal;
- sinc DVR in R1 and R2, a Legendre DVR in cos θ;
- lowest eigenvalues by sparse Lanczos (scipy);
- MACE energies on the grid from the same model and dtype as the client, through a single-molecule
  call.

Self-checks:
- grid convergence of the levels to 0.1 cm⁻¹;
- the harmonic frequencies from the MACE Hessian at the minimum against the DVR's harmonic limit
  (the same DVR on the quadratic expansion of the surface) to 0.1 cm⁻¹;
- the separable 1D limit against tier-1's `exact.py` (a Morse stretch with a frozen bend).

Band positions are compared with the band origins. A rotational contour shifts a band's first
moment away from its origin; tier 1 measured +15 cm⁻¹ (300 K) and +29 cm⁻¹ (100 K) for its
rotating oscillator. So **position deviations under 30 cm⁻¹ are reported but not classified as
failures**.

The stretch reference is the two stretch origins weighted by their harmonic weight in the Y-4
atomic C_vv: mode k weight ∝ Σ_i L_ik²/m_i, from the mass-weighted Hessian eigenvectors L. ν1 and
ν3 are about 100 cm⁻¹ apart and merge in broadened spectra. Where a spectrum resolves the two
peaks, they are compared separately.

Experimental gas-phase water (ν1 3657, ν2 1595, ν3 3756 cm⁻¹) is quoted for context only. The
surface's own error is not a dynamics failure.

### Protocol

- Temperatures 300 K and 150 K. The lower one is where tier 1's PA-CMD effects grew; 100 K would
  need P ≈ 192 and a PA-CMD Δt of ≈ 0.03 fs on this budget.
- P from the statics, as in tier 1: the virial energy within 1 % of H4's value. Expected P = 64 at
  300 K and 128 at 150 K.
- Δt: RPMD/TRPMD/classical start at 0.25 fs, checked against 0.125 fs in the pilot (C_vv band
  centres within σ). PA-CMD uses Ω = 16 000 cm⁻¹ and Δt from tier 1's rule, ω_e·Ω/ω_1·Δt ≈ 0.86 with
  ω_e the harmonic antisymmetric stretch: ≈ 0.1 fs at 300 K and ≈ 0.05 fs at 150 K. The pilot
  confirms that ⟨V⟩ is right at that Δt.
- Methods: classical (P = 1), RPMD, TRPMD (λ = 0.5), PA-CMD (Ω = 16 000 cm⁻¹, `cmd_lambda` = 1.0).
- Segments as tier 1: 100 fs decorrelation, 2 ps segments, lag range 2 ps on a lag grid of
  0.5 fs (`corr_stride` × Δt), n_lags = 4 001; 10 segments per run; replicas (independent
  seeds) set from the pilot's σ, at least 2.
- Q4: P × 2 (RPMD, PA-CMD), Δt / 2 (RPMD), Ω = 32 000 cm⁻¹ (PA-CMD), at 300 K, one replica each
  unless σ needs two. The Δt / 2 runs keep the same lag grid. Tier 1's 1 ps slip came from not
  doing this.
- Seeds fixed per run and logged; every config copied to `configs/tier2/`.

### Analysis (`analyse.py`, extended)

- Same transform as tier 1: Hann window over 2 ps, trapezoid rule.
- Per band, with windows set from the reference before any run (bend: ν2 ± 300 cm⁻¹; stretch:
  ν1 − 300 to ν3 + 300 cm⁻¹):
  - first-moment centre and standard-deviation width;
  - peak, refined with a guard for edge maxima;
  - the full-range centre over ω ≥ 1000 cm⁻¹ with each band's share, in case a band moves out of
    its window (tier 1's 100 K PA-CMD band did).
- Spurious weight: fraction of ∫|I| over ω ≥ 1000 cm⁻¹ lying more than 300 cm⁻¹ from every band
  origin and from 2ν2. No exact spectrum exists to subtract here, so this counts every feature
  away from the bands, including overtones and combinations the methods might legitimately show.
  2ν2 is excluded for that reason; any other feature is reported with its frequency against the
  internal-mode frequencies ω_k.
- σ: standard error over replicas.
- Statics: virial and primitive energy per molecule against E_ZPE + 3k_BT (H4), with block σ.

### What counts as a finding

- A **failure** is a deviation from the band origin larger than 30 cm⁻¹ (the contour allowance)
  plus 5σ plus the largest Q4 sensitivity.
- **Repeated** means seen at both temperatures, or in both bands, with the same sign.
- A static quantity outside 5σ of H4's value stops the series for a log entry and an owner
  decision, as in tier 1.

### Budget and resources

- **Machine:** the GPU (RTX 4060, 8 GB) runs one study run at a time, with K socket clients on it.
  CPU-only runs (e.g. classical) may run beside it.
- **Cost model:** one request per bead per step. M10 Q5 measured 49.5 ms per P = 32 step for
  one molecule on CUDA, a per-call floor. Batching N molecules per request shares that floor, so
  the pilot measures the time per request against N and K.
- **Estimate:** if a request costs ≈ 5 ms, a run at P = 64, Δt = 0.25 fs (≈ 90 k steps) takes
  ≈ 8 h at K = 1, and the matrix (2 T × 4 methods × 2 replicas plus Q4) would be far over budget.
  The pilot therefore fixes N, K, the replica count and the segment count to fit the cap.
- **Cap:** 72 h of wall time for tier 2. If the pilot projects more, stop and report with options
  (fewer methods at 150 K, fewer segments, or a CPU-parallel route).
- **Storage:** as tier 1 (`trajectory_every` large); outputs under `out/tier2/` (gitignored);
  results in `results/tier2/`.

### Pilot (first measurement step)

1. Client checks (finite differences, M10a identity, no cross terms); reference self-checks and
   band origins.
2. Time per request against N (16, 64, 256) and K (1, 2, 4) on CUDA, e3nn f64 (cueq/f32 only if
   needed); the budget follows.
3. Statics: P scan at both T against H4; Δt check (0.25 vs 0.125 fs) for RPMD; PA-CMD ⟨V⟩ at its
   Δt.
4. One short run per method at 300 K (a few segments): outputs, σ, run count.

The pilot's numbers go into the log. The production matrix is adjusted only by the rules above
(P, Δt, N, K, segments, replicas); any other change is logged with the reason before production.

### What would change the plan

- The reference fails its self-checks: fix it before any comparison.
- The budget exceeds the cap: report and ask (options above).
- PA-CMD not adiabatic at 16 000 cm⁻¹ (centres differ from 32 000 cm⁻¹ by more than the
  allowance): report and ask, rather than rerun silently (tier 1's open item).

## Log

### 2026-11-16 Pilot step 1: client and reference (GPU, one job at a time)

- `water_client.py --check` (CPU and CUDA, f64): one molecule identical to the M10a client (energy
  −76.47275116977178 Eh both); forces against central differences (h = 1e-4 bohr) 5.0e-8
  relative; four molecules with two overlapping (1.5 bohr apart) against the sum of single
  molecules |ΔE| = 1.7e-13 Eh, |ΔF| = 3.5e-17 Eh/bohr (no cross terms).
- `water_ref.py --check` (CUDA f64, `results/tier2/reference.json`):
  - Minimum: E = −76.4797923081 Eh, OH 1.81181 bohr, HOH 104.953°.
  - Harmonic frequencies, Cartesian 1632.66 / 3844.42 / 3948.60 cm⁻¹; Wilson GF in Radau
    coordinates 1632.65 / 3844.45 / 3948.64 (check (a): the Radau construction and masses).
  - Check (b), the kinetic operator: the DVR on the Radau-quadratic surface scaled by λ = 16, 64,
    256, on grids shrunk by λ^(−1/4), deviates from GF by δ = 1.86 / 0.94 / 0.47 cm⁻¹ (bend) and
    falls as 1/√λ. The extrapolated δ∞ is 0.000 for all three modes.
  - Grid: 32 × 32 × 40 against 40 × 40 × 48 moves levels ≤ 0.030 cm⁻¹; a wider r range
    (−0.75 / +1.4 bohr) ≤ 0.047 cm⁻¹.
  - Plan deviation in the check: the plan's "harmonic limit to 0.1 cm⁻¹ at fixed λ" was replaced
    by the λ → ∞ extrapolation. At finite λ the curvilinear kinetic energy leaves O(1/√λ)
    anharmonic terms, so a fixed-λ comparison is not a test. The first attempts at fixed λ = 400
    and with a 3σ-truncated r range failed for those reasons, not because of the operator.
  - **The surface has holes.** At H–H < 1 bohr (Radau θ ≲ 40°) MACE returns energies down to
    −368 Eh below the minimum. The physical surface there is > 24 000 cm⁻¹ above the minimum
    (V at the neighbouring θ = 46°). The reference sets V = 0.5 Eh (≈ 110 000 cm⁻¹) where
    H–H < 1.2 bohr and caps V at 0.5 Eh elsewhere. The lowest 12 states carry ≤ 1.4e-8 of their
    weight on capped points (a shift ≤ 0.002 cm⁻¹). PIMD never comes near these geometries at
    150–300 K, but the dynamics runs are watched for H–H < 1.2 bohr in case.
  - **Band origins (cm⁻¹; experiment in brackets):**

    | Level | MACE-OFF23 small | Experiment |
    |---|---|---|
    | ν2 | 1560.63 | 1594.7 |
    | 2ν2 | 3074.34 | 3151.6 |
    | ν1 | 3679.82 | 3657.1 |
    | ν3 | 3778.35 | 3755.9 |

    ZPE = 4628.82 cm⁻¹ (0.02109044 Eh). The Y-4 weights of bend / ν1 / ν3 in the atomic C_vv are
    0.329 / 0.341 / 0.329, so the stretch reference centre is 3728.2 cm⁻¹.
  - H4 values: E_min + ZPE + 3k_BT = −76.4558517 Eh per molecule at 300 K and −76.4572768 at
    150 K (−25.4852839 and −25.4857589 Eh per atom, the engine's normalisation). Vibrational
    excitation adds 4e-6 Eh per molecule at 300 K, which the statics script includes.

### 2026-11-16 Pilot step 2: cost per step, and a batched client

- Through the socket, one request per bead per step. CUDA, e3nn, per request:

  | Molecules | f64 | f32 |
  |---|---|---|
  | 1 | 41 ms | 34 ms |
  | 16 | 32 ms | 37 ms |
  | 64 | 86 ms | 41 ms |
  | 256 | 307 ms | 36 ms |

  The f32 numbers are a per-call floor. CPU f64, one thread: 11.5 / 107 / 474 / 2112 ms. At P = 64
  this would make a run ≈ 14 h even with K = 4: over the cap.
- **Batched client (`water_mux.py`, study-only, no engine change).** The engine's socket potential
  runs one thread per client, each STATUS → POSDATA + GETFORCE (crates/pimd-socket/src/lib.rs). With
  `n_clients = P`, one process holds P connections, collects every bead's positions, and evaluates
  all P × N molecules in one MACE call. Each connection gets its bead's energy (per-node
  interaction energies plus Σ E0 in f64) and forces.
  - Check against `water_client.py` bead by bead: f64 |ΔE| 1.7e-13 Eh, |ΔF| 4e-17 Eh/bohr; f32
    6e-8 Eh, 2e-8 Eh/bohr; cueq f32 (`.venv-cueq`, inside `pixi run -e ml` for Triton's ptxas)
    against e3nn f32 1.2e-7 Eh, 2.8e-8 Eh/bohr.
- Engine ms per step (50-step difference of static runs; batched; CUDA):

  | Route | N = 8 | N = 16 | N = 24 | N = 32 | N = 64 |
  |---|---|---|---|---|---|
  | e3nn f32, P = 64 | 58.7 | 110.7 | 167.6 | 221.1 | 2 822 (spills past 8 GB) |
  | e3nn f32, P = 128 | 122.7 | 232.2 | | | |
  | e3nn f64, P = 64 | | 1 224 | | | out of memory |
  | cueq f32, P = 64 | | 30.9 | | 53.5 | 97.9 |
  | cueq f32, P = 128 | | | | 102.8 | |

  e3nn f32 costs ≈ 0.11 ms per molecule-bead, cueq f32 ≈ 0.024 ms, and e3nn f64 ≈ 1.2 ms (FP64 on
  this consumer GPU). **f64 does not fit:** a P = 64, N = 16 run is ≈ 29 h.
- Route for the rest of the pilot: batched cueq f32, N = 64. This is subject to the plan's condition
  that f32/cueq keeps NVE RPMD drift and band positions within σ of f64. It is checked first in step
  3, against e3nn f64 at sizes f64 can afford (classical P = 1, N = 64; RPMD P = 64, N = 4).

### 2026-11-16 Pilot steps 3–4: precision, statics, PA-CMD stability, short dynamics

Queue `pilot_t2.sh`, batched client, N = 64 unless stated, one GPU job at a time, 10.1 h of wall
time (01:25–11:19). Configs in `studies/m11-dynamics-failure/configs/tier2/pilot_*.toml`; numbers
from `statics_t2.py` (results/tier2/pilot_statics.txt) and `analyse_t2.py`
(results/tier2/pilot_spectra.json). One replica per group unless stated, so peaks are noisy; the
band centres are the metric used below.

**3a/3b. Precision (cueq f32 against e3nn f64, same seeds).** Passes.

| Check | cueq f32 | e3nn f64 |
|---|---|---|
| Classical 300 K, 5 segments: bend / stretch centre (cm⁻¹) | 1615.7 / 3859.1 | 1615.5 / 3859.2 |
| Classical: largest \|Δh\| per 2 ps NVE segment (Eh per atom) | 2.8e-7 | 4.0e-7 |
| RPMD P = 64, N = 4, one 2 ps NVE segment: Δh (Eh, of −1619.43) | +1.720e-3 | +1.709e-3 |

The drift is the integrator's, not the precision's. Production route: batched cueq f32.

**3c. Statics against H4** (virial and primitive energy above E_min per molecule; σ from 10 blocks):

| T | P | E_vir dev. | E_prim dev. |
|---|---|---|---|
| 300 K | 16 | −10.14 % ± 0.13 | −10.35 % ± 0.19 |
| 300 K | 32 | −3.01 % ± 0.07 | −3.47 % ± 0.10 |
| 300 K | 64 | −0.52 % ± 0.17 | −1.03 % ± 0.33 |
| 300 K | 128 | +0.10 % ± 0.14 | −1.78 % ± 0.25 |
| 150 K | 32 | −10.91 % ± 0.09 | −10.65 % ± 0.13 |
| 150 K | 64 | −3.08 % ± 0.10 | −3.23 % ± 0.13 |
| 150 K | 128 | −0.82 % ± 0.09 | −1.60 % ± 0.24 |
| 150 K | 256 | −0.22 % ± 0.11 | −2.03 % ± 0.29 |

- The virial estimator converges with P as expected. By the plan's rule (virial within 1 %),
  production P is 64 at 300 K and 128 at 150 K (as the plan expected).
- **Primitive − virial grows with P and does not average away.** In Eh per molecule, with σ from
  the integrated autocorrelation time:
  - −0.5e-4 (P = 16) and −1.2e-4 (P = 64) at 300 K;
  - −4.5e-4 ± 0.6e-4 at 300 K, P = 128 (7σ);
  - −4.1e-4 ± 0.5e-4 at 150 K, P = 256 (8σ).

  Both halves of each run agree. The two estimators have the same expectation in the finite-P
  ensemble, so the sampling is biased in a way the virial estimator hides.
  - A candidate, not tested: the configurational Δt error of a splitting with the exact free
    ring-polymer step, which grows with P (Korol, Bou-Rabee and Miller 2019, the reason for the
    Cayley-modified step).
  - Other candidates: f32 forces, or the thermostat.
  - Test: one static run at Δt = 0.125 fs, T = 300 K, P = 128 (≈ 40 min). If the gap is a Δt
    error, it should shrink about 4×.
- **This triggers the plan's statics rule** (a static quantity outside 5σ of H4 stops the series
  for a log entry and an owner decision). At the production P values the primitive estimator is
  off by 1.0 % (300 K, P = 64; 3σ on 10 blocks) and 1.6 % (150 K, P = 128; 7σ).

**3d. PA-CMD at 150 K (P = 128, Δt = 0.05 fs, Ω = 16 000 cm⁻¹, one 1 ps segment).**
- Stable, with every energy finite.
- ⟨V⟩ − E_min per molecule is 0.010687 ± 0.000008 Eh, against plain PIMD at the same P
  (T150_P128) 0.010656 ± 0.000012: 0.3 %, about 2σ.
- Tier 1's silent failure (⟨V⟩ 24× exact) is absent at this Δt.

**3e. Δt check (RPMD, 300 K, P = 64, 3 segments, same 0.5 fs lag grid).**
- Bend centre: 1546.2 at 0.25 fs against 1543.1 cm⁻¹ at 0.125 fs (Δ 3.1).
- Stretch centre: 3768.9 against 3768.0 cm⁻¹ (Δ 0.9).
- The plan asks for "within σ". σ of an RPMD centre is not known from one replica each. Two
  classical replicas (prec/classical_cueq and dyn/classical; 5 and 3 segments) give σ of 0.3 (bend)
  and < 0.1 (stretch) cm⁻¹. On that scale the bend difference is about 10σ, though still a tenth
  of the 30 cm⁻¹ allowance. The Q4 Δt/2 run would carry it into the finding threshold.

**4. Short runs, 300 K, P = 64, 3 segments, one replica** (centres in cm⁻¹; reference ν2 1560.6,
stretch 3728.2):

| Method | Bend | Stretch | Spurious | ms per step |
|---|---|---|---|---|
| Classical (P = 1) | 1616.4 | 3859.1 | 0.032 | 36.7 |
| RPMD | 1546.2 | 3768.9 | 0.056 | 98.2 |
| TRPMD | 1550.3 | 3739.0 | 0.065 | 98.2 |
| PA-CMD (Δt 0.1 fs) | 1544.7 | 3626.4 | 0.029 | 97.5 |

These are pilot numbers for sizing only, not results. The minimum V − E_min per molecule is
≥ +9.7e-4 Eh in every run, so no molecule entered a hole. Cost at P = 128: 193.5 ms per step
(191 for PA-CMD).

**Production cost** (from the measured ms per step; 2 replicas per main group; Q4 at 300 K one
run each: RPMD P = 128, PA-CMD P = 128, RPMD Δt/2, and PA-CMD Ω = 32 000 at Δt 0.05 fs):

| Segments per run | Main matrix | of which PA-CMD 150 K | Q4 | Total |
|---|---|---|---|---|
| 10 (plan) | 87.5 h | 45.0 h | 32.0 h | 119.5 h |
| 5 | 44.1 h | 22.7 h | 16.2 h | 60.3 h |
| 4 | 35.5 h | 18.3 h | 13.0 h | 48.5 h |

About 61 h of the 72 h cap is left after the pilot. The plan's matrix is **over the cap** ("report
and ask"). With 64 molecules per run, σ of a centre is already well under 1 cm⁻¹ at 3–5 segments,
so fewer segments cost little precision against the 30 cm⁻¹ allowance.

Production does not start until the owner decides on: the primitive-estimator gap (3c), the Δt
check (3e), and the segment count (budget).
