# M11 study, tier 2: dynamics failure modes in gas-phase H₂O (MACE-OFF23 small)

Status: **plan (draft, 2026-11-15); awaiting owner approval.** Experiment flow: this plan is
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

(empty: no measurement yet)
