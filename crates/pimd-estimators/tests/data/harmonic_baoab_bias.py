"""Exact time-step bias of BAOAB + PILE-L for the harmonic ring polymer (tolerance derivation only).

Run: pixi run -e ref python crates/pimd-estimators/tests/data/harmonic_baoab_bias.py

Not reference data: it prints the discretisation-bias table quoted in harmonic_sampling.rs.
For `Harmonic` the external force -m w^2 q is diagonal in the RP-4 normal modes, so each mode and
Cartesian component is an independent linear stochastic map x' = F x + G xi with
F = B A O A B (SPEC I-1, I-2, T-1, T-2), B = [[1, 0], [-(dt/2) m w^2, 1]], A the exact rotation at
w_k for dt/2 (drift for w_k = 0), O = diag(1, c_k). Its stationary covariance solves the discrete
Lyapunov equation S = F S F^T + G G^T, which gives <q_k^2> exactly at finite dt. The estimators are
linear in <q_k^2>: per atom and component
    E_prim = P/(2 beta) + sum_k (w^2 - w_k^2) <m q_k^2> / (2P),
    E_vir  = 1/(2 beta) + sum_k w^2 <m q_k^2> / (2P) + sum_{k>=1} w^2 <m q_k^2> / (2P),
and at dt -> 0, <m q_k^2> = P / (beta (w_k^2 + w^2)) reproduces SPEC E-9.
"""

import numpy as np
import scipy.linalg as sl


def mode_cov(wk, w, beta, p, dt, gamma, m=1.0):
    b = np.array([[1.0, 0.0], [-0.5 * dt * m * w * w, 1.0]])
    t = dt / 2
    if wk > 0:
        a = np.array([[np.cos(wk * t), np.sin(wk * t) / (m * wk)],
                      [-m * wk * np.sin(wk * t), np.cos(wk * t)]])
    else:
        a = np.array([[1.0, t / m], [0.0, 1.0]])
    c = np.exp(-gamma * dt)
    o = np.diag([1.0, c])
    s = np.sqrt((1 - c * c) * m * p / beta)
    f = b @ a @ o @ a @ b
    g = b @ a @ np.array([[0.0], [s]])
    return sl.solve_discrete_lyapunov(f, g @ g.T)


def energies(p, dt, beta, w, tau0, n_atoms):
    wp = p / beta
    wk = [2 * wp * np.sin(np.pi * k / p) for k in range(p)]
    dof = 3 * n_atoms
    eprim, evir, exact = dof * p / (2 * beta), dof / (2 * beta), 0.0
    for k in range(p):
        gamma = 1 / tau0 if k == 0 else 2 * wk[k]
        q2 = mode_cov(wk[k], w, beta, p, dt, gamma)[0, 0]
        exact += dof / beta * w * w / (wk[k] ** 2 + w * w)
        eprim += dof * (w * w - wk[k] ** 2) * q2 / (2 * p)
        evir += dof * w * w * q2 / (2 * p) * (2 if k > 0 else 1)
    return eprim, evir, exact


if __name__ == "__main__":
    # Settings of harmonic_sampling.rs: N = 4, w = 1, beta = 4, tau0 = 1.
    for p in [1, 4, 16]:
        for dt in [0.05, 0.025]:
            ep, ev, ex = energies(p, dt, 4.0, 1.0, 1.0, 4)
            print(f"P={p:2d} dt={dt}: E-9 {ex:.6f}  bias E_prim {ep - ex:+.2e}  E_vir {ev - ex:+.2e}")
