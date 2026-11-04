//! Ring-polymer state, BAOAB integrator with PILE-L and TRPMD (SPEC I-1 … I-5, T-1 … T-4, Y-1, Y-2).

use pimd_core::{Beads, Cell, NormalModes, Rng, Transform};
use pimd_potentials::{Evaluation, Potential};

use crate::error::IntegratorError;

/// Ring-polymer phase-space point in the bead representation (RP-3, S-1).
#[derive(Debug, Clone, PartialEq)]
pub struct RingState {
    /// Bead positions, shape `[P, N, 3]`, in bohr.
    pub q: Beads,
    /// Bead momenta, shape `[P, N, 3]`, in m_e·bohr per ħ/Eh (atomic units). Their variance in
    /// equilibrium is m_i·P/β per component (RP-3).
    pub p: Beads,
    /// One mass per atom, length N, in m_e (S-2).
    pub masses: Vec<f64>,
}

/// Thermostat applied in the O step (T-1, T-2, T-4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Thermostat {
    /// No thermostat: the O step is the identity (T-3).
    Off,
    /// PILE-L (T-1). `tau0` is the centroid time constant in ħ/Eh, > 0 when `Some`;
    /// `None` leaves the centroid unthermostatted (γ_0 = 0, T-2).
    PileL {
        /// Centroid time constant τ_0 in ħ/Eh, or `None` for γ_0 = 0.
        tau0: Option<f64>,
    },
    /// Thermostatted RPMD (T-4): γ_0 = 0 and γ_k = 2·`lambda`·ω_k for every mode with ω_k > 0;
    /// draw order and count as T-1. `lambda` > 0 and finite.
    Trpmd {
        /// Scaling λ of the critical damping 2ω_k of the internal modes (dimensionless).
        lambda: f64,
    },
}

/// BAOAB ring-polymer integrator with exact free propagation and optional PILE-L (I-1, I-2, T-1).
#[derive(Debug, Clone)]
pub struct RingPolymerIntegrator {
    /// Time step, ħ/Eh.
    dt: f64,
    /// Inverse temperature, Eh⁻¹.
    beta: f64,
    modes: NormalModes,
    /// c_k = exp(−γ_k dt) per mode, aligned with the `modes` columns (T-1, T-2);
    /// `None` for `Thermostat::Off`, whose O step is the identity and draws nothing (T-3).
    c_k: Option<Vec<f64>>,
}

/// Returns `v` if it is positive and finite, else `IntegratorError::InvalidValue` named `what`.
pub(crate) fn positive_finite(what: &'static str, v: f64) -> Result<f64, IntegratorError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(IntegratorError::InvalidValue { what, value: v })
    }
}

impl RingPolymerIntegrator {
    /// Integrator with time step `dt` (ħ/Eh), inverse temperature `beta` (Eh⁻¹) and `n_beads` = P.
    ///
    /// Errors: `IntegratorError::InvalidValue` unless dt (I-5) and β are positive and finite, and
    /// τ_0 (T-2) is positive and finite when given, and λ (T-4) is positive and finite;
    /// `IntegratorError::Core` for P = 0.
    pub fn new(
        dt: f64,
        beta: f64,
        n_beads: usize,
        thermostat: Thermostat,
    ) -> Result<Self, IntegratorError> {
        Self::with_transform(dt, beta, n_beads, thermostat, Transform::Matrix)
    }

    /// As `new`, with the normal-mode transform algorithm `transform` (RP-4). `new(..)` is
    /// `with_transform(.., Transform::Matrix)`. The mode order, frequencies, c_k and the
    /// thermostat draw order (T-1) do not depend on `transform`.
    ///
    /// Errors: as `new`.
    pub fn with_transform(
        dt: f64,
        beta: f64,
        n_beads: usize,
        thermostat: Thermostat,
        transform: Transform,
    ) -> Result<Self, IntegratorError> {
        positive_finite("dt (hbar/Eh)", dt)?;
        positive_finite("beta (1/Eh)", beta)?;
        let gammas = friction(thermostat)?;
        let modes = NormalModes::with_transform(n_beads, beta, transform)?;
        let c_k = gammas.map(|g| ou_coefficients(g, &modes, dt));
        Ok(RingPolymerIntegrator {
            dt,
            beta,
            modes,
            c_k,
        })
    }

    /// Exact free ring-polymer propagation of `state` for time `t` (ħ/Eh, any sign), per normal
    /// mode and component (I-2), using `state.masses`. No potential forces.
    ///
    /// `state.masses` must be positive and finite (S-2); this is not checked here
    /// (`RingPolymerSim::new` checks it).
    ///
    /// Panics if the bead count of `state` differs from P or `state.masses.len()` from N.
    pub fn free_propagate(&self, state: &mut RingState, t: f64) {
        self.assert_shape(state);
        let mut qt = self.modes.to_normal(&state.q);
        let mut pt = self.modes.to_normal(&state.p);
        self.rotate_modes(&mut qt, &mut pt, &state.masses, t);
        state.q = self.modes.to_beads(&qt);
        state.p = self.modes.to_beads(&pt);
    }

    /// One O step of length dt (T-1): PILE-L or TRPMD (T-4) on the normal-mode momenta, or the
    /// identity for `Thermostat::Off`. Draws one normal per (k, i, d), k outermost, then atom i,
    /// then d.
    /// `Thermostat::Off` draws nothing (T-3).
    ///
    /// `state.masses` must be positive and finite (S-2); this is not checked here
    /// (`RingPolymerSim::new` checks it).
    ///
    /// Panics if the bead count of `state` differs from P or `state.masses.len()` from N.
    pub fn thermostat_step(&self, state: &mut RingState, rng: &mut Rng) {
        self.assert_shape(state);
        if self.c_k.is_none() {
            return;
        }
        let mut pt = self.modes.to_normal(&state.p);
        self.thermostat_modes(&mut pt, &state.masses, rng);
        state.p = self.modes.to_beads(&pt);
    }

    /// The normal-mode transform used by this integrator (RP-4).
    pub fn modes(&self) -> &NormalModes {
        &self.modes
    }

    /// `IntegratorError::ShapeMismatch` unless q, p and masses agree with each other and with P.
    fn check_shape(&self, state: &RingState) -> Result<(), IntegratorError> {
        let (p, n) = (state.q.n_beads(), state.q.n_atoms());
        check_sizes(&[
            ("bead count of q vs integrator", self.modes.n_beads(), p),
            ("bead count of p", p, state.p.n_beads()),
            ("atom count of p", n, state.p.n_atoms()),
            ("number of masses", n, state.masses.len()),
        ])
    }

    fn assert_shape(&self, state: &RingState) {
        if let Err(e) = self.check_shape(state) {
            panic!("{e}");
        }
    }

    /// I-2 in normal-mode coordinates `qt`, `pt` (`[P, N, 3]`, mode-major) for time `t`.
    fn rotate_modes(&self, qt: &mut Beads, pt: &mut Beads, masses: &[f64], t: f64) {
        let n3 = masses.len() * 3;
        let (qs, ps) = (qt.as_mut_slice(), pt.as_mut_slice());
        for (k, &w) in self.modes.frequencies().iter().enumerate() {
            let mode = k * n3..(k + 1) * n3;
            let pairs = qs[mode.clone()]
                .chunks_exact_mut(3)
                .zip(ps[mode].chunks_exact_mut(3));
            if w > 0.0 {
                let (s, c) = (w * t).sin_cos();
                for ((q, p), &m) in pairs.zip(masses) {
                    for (q, p) in q.iter_mut().zip(p) {
                        let (q0, p0) = (*q, *p);
                        *q = c * q0 + s / (m * w) * p0;
                        *p = -m * w * s * q0 + c * p0;
                    }
                }
            } else {
                for ((q, p), &m) in pairs.zip(masses) {
                    for (q, p) in q.iter_mut().zip(p.iter()) {
                        *q += t * p / m;
                    }
                }
            }
        }
    }

    /// T-1 on normal-mode momenta `pt`; draws k outermost, then atom i, then d. No-op for `Off`.
    fn thermostat_modes(&self, pt: &mut Beads, masses: &[f64], rng: &mut Rng) {
        let Some(c_k) = &self.c_k else {
            return;
        };
        let pf = self.modes.n_beads() as f64;
        let n3 = masses.len() * 3;
        let ps = pt.as_mut_slice();
        for (k, &c) in c_k.iter().enumerate() {
            for (p, &m) in ps[k * n3..(k + 1) * n3].chunks_exact_mut(3).zip(masses) {
                let amp = ((1.0 - c * c) * m * pf / self.beta).sqrt();
                for x in p {
                    *x = c * *x + amp * rng.normal();
                }
            }
        }
    }
}

/// Friction rule (γ_0, scale of ω_k for k ≥ 1) of `thermostat`, or `None` for `Off`:
/// T-2 for PILE-L (γ_0 = 1/τ_0 or 0, γ_k = 2ω_k), T-4 for TRPMD (γ_0 = 0, γ_k = 2λω_k).
///
/// Errors: `IntegratorError::InvalidValue` for τ_0 or λ not positive and finite.
fn friction(thermostat: Thermostat) -> Result<Option<(f64, f64)>, IntegratorError> {
    Ok(match thermostat {
        Thermostat::Off => None,
        Thermostat::PileL { tau0: None } => Some((0.0, 2.0)),
        Thermostat::PileL { tau0: Some(tau0) } => {
            Some((1.0 / positive_finite("tau0 (hbar/Eh)", tau0)?, 2.0))
        }
        Thermostat::Trpmd { lambda } => Some((0.0, 2.0 * positive_finite("lambda", lambda)?)),
    })
}

/// c_k = exp(−γ_k dt) per mode (T-1), aligned with the `modes` columns, for the friction rule
/// `(γ_0, s)` of `friction`: γ_0 for k = 0, γ_k = s·ω_k otherwise.
fn ou_coefficients((gamma0, s): (f64, f64), modes: &NormalModes, dt: f64) -> Vec<f64> {
    modes
        .frequencies()
        .iter()
        .enumerate()
        .map(|(k, &w)| {
            let gamma = if k == 0 { gamma0 } else { s * w };
            (-gamma * dt).exp()
        })
        .collect()
}

/// `IntegratorError::ShapeMismatch` unless `ev` has forces of shape `[p, n, 3]` and `p`
/// energies (P-1).
/// The construction checks shared by `new` and `from_parts`: shapes of `state` against the
/// integrator (P-1) and masses positive and finite (S-2). Returns (P, N).
fn check_state(
    integrator: &RingPolymerIntegrator,
    state: &RingState,
) -> Result<(usize, usize), IntegratorError> {
    integrator.check_shape(state)?;
    for &m in &state.masses {
        positive_finite("mass (m_e)", m)?;
    }
    Ok((state.q.n_beads(), state.q.n_atoms()))
}

pub(crate) fn check_evaluation(ev: &Evaluation, p: usize, n: usize) -> Result<(), IntegratorError> {
    check_sizes(&[
        ("bead count of evaluated forces", p, ev.forces.n_beads()),
        ("atom count of evaluated forces", n, ev.forces.n_atoms()),
        ("number of evaluated energies", p, ev.energies.len()),
    ])
}

/// The first `(what, expected, actual)` with `expected != actual`, as `ShapeMismatch`.
pub(crate) fn check_sizes(checks: &[(&'static str, usize, usize)]) -> Result<(), IntegratorError> {
    for &(what, expected, actual) in checks {
        if expected != actual {
            return Err(IntegratorError::ShapeMismatch {
                what,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

/// p ← p + h·F, element-wise (the B step of I-1 with h = dt/2).
pub(crate) fn kick(p: &mut Beads, f: &Beads, h: f64) {
    for (p, f) in p.as_mut_slice().iter_mut().zip(f.as_slice()) {
        *p += h * f;
    }
}

/// A ring-polymer simulation: state, current forces and energies, integrator and potential.
pub struct RingPolymerSim {
    state: RingState,
    forces: Beads,
    energies: Vec<f64>,
    integrator: RingPolymerIntegrator,
    potential: Box<dyn Potential>,
    cell: Option<Cell>,
}

impl RingPolymerSim {
    /// Builds the simulation and performs the initialising `evaluate` (I-3).
    ///
    /// Errors: `IntegratorError::ShapeMismatch` if `state.q`, `state.p`, `state.masses` and the
    /// integrator's P disagree, or if the evaluation's forces are not `[P, N, 3]` or its energies
    /// not length P (P-1); `IntegratorError::InvalidValue` if a mass is not positive and finite
    /// (S-2); `IntegratorError::Potential` if the evaluation fails.
    pub fn new(
        state: RingState,
        cell: Option<Cell>,
        integrator: RingPolymerIntegrator,
        mut potential: Box<dyn Potential>,
    ) -> Result<Self, IntegratorError> {
        let (p, n) = check_state(&integrator, &state)?;
        let ev = potential.evaluate(&state.q, cell.as_ref(), None)?;
        check_evaluation(&ev, p, n)?;
        Ok(RingPolymerSim {
            state,
            forces: ev.forces,
            energies: ev.energies,
            integrator,
            potential,
            cell,
        })
    }

    /// As `new`, but takes the forces (`[P, N, 3]`, Eh/bohr) and per-bead energies (length P, Eh)
    /// at `state.q` instead of evaluating them (restart, Q18). Validates masses (S-2) and shapes
    /// (P-1) like `new`; no `evaluate` call.
    ///
    /// Errors: as `new`, except that no evaluation can fail: `IntegratorError::ShapeMismatch` if
    /// `state`, the integrator's P, `forces` or `energies` disagree in size;
    /// `IntegratorError::InvalidValue` if a mass is not positive and finite, or a given force
    /// (`what: "given forces"`) or energy (`what: "given energies"`) is not finite (Q23).
    pub fn from_parts(
        state: RingState,
        cell: Option<Cell>,
        integrator: RingPolymerIntegrator,
        potential: Box<dyn Potential>,
        forces: Beads,
        energies: Vec<f64>,
    ) -> Result<Self, IntegratorError> {
        let (p, n) = check_state(&integrator, &state)?;
        check_sizes(&[
            ("bead count of given forces", p, forces.n_beads()),
            ("atom count of given forces", n, forces.n_atoms()),
            ("number of given energies", p, energies.len()),
        ])?;
        for (what, values) in [
            ("given forces", forces.as_slice()),
            ("given energies", energies.as_slice()),
        ] {
            if let Some(&value) = values.iter().find(|v| !v.is_finite()) {
                return Err(IntegratorError::InvalidValue { what, value });
            }
        }
        Ok(RingPolymerSim {
            state,
            forces,
            energies,
            integrator,
            potential,
            cell,
        })
    }

    /// One step B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2) (I-1), with exactly one `evaluate` (I-3).
    /// A O A run in normal-mode coordinates (one transform each way per step).
    ///
    /// Errors: `IntegratorError::Potential` if the evaluation fails, or
    /// `IntegratorError::ShapeMismatch` if its forces are not `[P, N, 3]` or its energies not
    /// length P (P-1). The simulation is then left unchanged, but the caller's `rng` has already
    /// advanced: P·N·3 normal draws under PILE-L and TRPMD, none under `Off` (D-4).
    pub fn step(&mut self, rng: &mut Rng) -> Result<(), IntegratorError> {
        let ig = &self.integrator;
        let half = 0.5 * ig.dt;
        let masses = &self.state.masses;
        let mut p = self.state.p.clone();
        kick(&mut p, &self.forces, half);
        let mut qt = ig.modes.to_normal(&self.state.q);
        let mut pt = ig.modes.to_normal(&p);
        ig.rotate_modes(&mut qt, &mut pt, masses, half);
        ig.thermostat_modes(&mut pt, masses, rng);
        ig.rotate_modes(&mut qt, &mut pt, masses, half);
        let q = ig.modes.to_beads(&qt);
        let mut p = ig.modes.to_beads(&pt);
        let ev = self.potential.evaluate(&q, self.cell.as_ref(), None)?;
        check_evaluation(&ev, q.n_beads(), q.n_atoms())?;
        kick(&mut p, &ev.forces, half);
        self.state.q = q;
        self.state.p = p;
        self.forces = ev.forces;
        self.energies = ev.energies;
        Ok(())
    }

    /// Replaces the O step's thermostat (Y-1). Keeps state, forces, energies, dt, β, P and the
    /// transform; no `evaluate`, no draws.
    ///
    /// Errors: `IntegratorError::InvalidValue` for a bad τ_0 or λ (as `RingPolymerIntegrator::new`);
    /// the simulation is then unchanged.
    pub fn set_thermostat(&mut self, thermostat: Thermostat) -> Result<(), IntegratorError> {
        let ig = &mut self.integrator;
        ig.c_k = friction(thermostat)?.map(|g| ou_coefficients(g, &ig.modes, ig.dt));
        Ok(())
    }

    /// Redraws every bead momentum from N(0, m_i·P/β) (Y-2): atom i outer, bead j, dimension d
    /// inner; P·N·3 normal draws. Positions, forces and energies unchanged.
    pub fn resample_momenta(&mut self, rng: &mut Rng) {
        let pf = self.integrator.modes.n_beads() as f64;
        let beta = self.integrator.beta;
        let p = &mut self.state.p;
        for (i, &m) in self.state.masses.iter().enumerate() {
            let sd = (m * pf / beta).sqrt();
            for j in 0..p.n_beads() {
                let v = [sd * rng.normal(), sd * rng.normal(), sd * rng.normal()];
                p.set(j, i, v);
            }
        }
    }

    /// The current state (bead representation).
    pub fn state(&self) -> &RingState {
        &self.state
    }

    /// Forces −∇V at the current positions, shape `[P, N, 3]`, Eh/bohr (S-3).
    pub fn forces(&self) -> &Beads {
        &self.forces
    }

    /// V(q_j) per bead at the current positions, length P, Eh (S-4).
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }
}
