//! Classical velocity Verlet for a single replica (SPEC I-4, I-5).

use pimd_core::{Beads, Cell};
use pimd_potentials::Potential;

use crate::error::IntegratorError;
use crate::ring_polymer::{RingState, check_evaluation, check_sizes, kick, positive_finite};

/// Standard velocity Verlet with time step `dt` in ħ/Eh (I-4), for states with P = 1.
#[derive(Debug, Clone)]
pub struct VelocityVerlet {
    dt: f64,
}

impl VelocityVerlet {
    /// Errors with `IntegratorError::InvalidValue` unless `dt` (ħ/Eh) is positive and finite (I-5).
    pub fn new(dt: f64) -> Result<VelocityVerlet, IntegratorError> {
        Ok(VelocityVerlet {
            dt: positive_finite("dt (hbar/Eh)", dt)?,
        })
    }

    /// One step: p ← p + (dt/2) F; q ← q + dt p/m; F ← F(q) (one `evaluate`); p ← p + (dt/2) F.
    ///
    /// `forces` holds F at the current positions on entry and F at the new positions on exit.
    /// Returns the potential energy at the new positions. Requires state with exactly one bead.
    /// Positions are never wrapped (C-5).
    ///
    /// Errors: `IntegratorError::ShapeMismatch` if the state has P ≠ 1, or `state.p`, `forces`,
    /// `state.masses` disagree with `state.q`; `IntegratorError::Potential` if the evaluation fails.
    pub fn step(
        &self,
        state: &mut RingState,
        forces: &mut Beads,
        potential: &mut dyn Potential,
        cell: Option<&Cell>,
    ) -> Result<f64, IntegratorError> {
        let n = state.q.n_atoms();
        check_sizes(&[
            (
                "bead count (velocity Verlet needs P = 1)",
                1,
                state.q.n_beads(),
            ),
            ("bead count of p", 1, state.p.n_beads()),
            ("atom count of p", n, state.p.n_atoms()),
            ("number of masses", n, state.masses.len()),
            ("bead count of forces", 1, forces.n_beads()),
            ("atom count of forces", n, forces.n_atoms()),
        ])?;
        let half = 0.5 * self.dt;
        let mut p = state.p.clone();
        kick(&mut p, forces, half);
        let mut q = state.q.clone();
        for (i, &m) in state.masses.iter().enumerate() {
            let (x, v) = (q.get(0, i), p.get(0, i));
            q.set(0, i, [0, 1, 2].map(|d| x[d] + self.dt * v[d] / m));
        }
        let ev = potential.evaluate(&q, cell, None)?;
        check_evaluation(&ev, 1, n)?;
        kick(&mut p, &ev.forces, half);
        state.q = q;
        state.p = p;
        *forces = ev.forces;
        Ok(ev.energies[0])
    }
}
