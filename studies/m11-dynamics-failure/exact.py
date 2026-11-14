"""Exact Kubo-transformed centroid correlation functions for the M11 tier-1 Morse systems.

    pixi run -e ref python studies/m11-dynamics-failure/exact.py --check
    (as a module: exact.reference(system, T_kelvin, times_au) -> dict)

Atomic units (ħ = 1). Kubo-transformed C̃_AB(t) = (1/(βZ)) Σ_{n,m} w_nm A_nm B_mn cos(ω_mn t), with
w_nm = (e^{-βE_n} - e^{-βE_m})/(E_m - E_n) (β e^{-βE_n} when E_m = E_n), and v_nm = i ω_nm x_nm, so
|v_nm|² = ω_nm² |x_nm|². Normalised as the engine's Y-4 estimator: per particle, summed over x, y, z.
  s1  separable 1D Morse per component: 3 × the 1D function (C_xx includes the constant ⟨x⟩²).
  s2  radial Morse V(|q|): radial sinc DVR per l, centrifugal l(l+1)/(2μr²); the dipole pairs
      (n l) ↔ (n' l±1) carry, summed over M, M' and components, the weight max(l, l')·|R|²,
      R = ∫ u_nl r u_n'l' dr; Z = Σ (2l+1) e^{-βE_nl}.
Mass and constants: the engine's U-2 values (pimd_core::units), copied.
"""

import argparse
import sys

import numpy as np

K_B = 3.1668115634556e-6  # Eh/K, U-2
AMU = 1822.888486209  # m_e, U-2
MU = 15.999 * 1.008 / (15.999 + 1.008) * AMU  # OH reduced mass (engine masses_amu), m_e
D, A, R_E = 0.18748, 1.1605, 1.8324  # Eh, 1/bohr, bohr (morse_client.py)
OMEGA_E = A * np.sqrt(2.0 * D / MU)
WEIGHT_CUT = 1e-12  # Boltzmann weight below which initial states are dropped


def morse(r):
    return D * (1.0 - np.exp(-A * (r - R_E))) ** 2


def sinc_dvr(x, v, mass):
    """Eigenpairs of -1/(2m) d²/dx² + v on a Colbert-Miller sinc DVR on the uniform grid x."""
    n, dx = len(x), x[1] - x[0]
    d = np.arange(n)[:, None] - np.arange(n)[None, :]
    with np.errstate(divide="ignore"):
        t = np.where(d == 0, np.pi**2 / 3.0, 2.0 / np.where(d == 0, 1, d) ** 2)
    t = t * (-1.0) ** np.abs(d) / (2.0 * mass * dx * dx)
    return np.linalg.eigh(t + np.diag(v))


def kubo_weights(beta, e_i, e_f, z):
    """(1/(βZ)) (e^{-βE_i} - e^{-βE_f})/(E_f - E_i) for every (i, f) pair; energies shifted."""
    de = e_f[None, :] - e_i[:, None]
    bi, bf = np.exp(-beta * e_i)[:, None], np.exp(-beta * e_f)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(de) > 1e-13, (bi - bf) / de, beta * bi)
    return w / (beta * z), de


def mirror(final_is_initial):
    """Pair multiplicity: the double sum over (n, m) counts each pair twice (w, |x|² symmetric).
    A pair whose final state is also a kept initial state is met again from that side; one whose
    final state is not is met once here and counted twice."""
    return np.where(final_is_initial, 1.0, 2.0)[None, :]


def lines_s1(beta, grid, potential=morse):
    """1D transition lines (frequency, C_xx amplitude) and statics, × 3 for the three components."""
    e, u = sinc_dvr(grid, potential(grid), MU)
    e = e - e[0]
    keep_i = np.exp(-beta * e) > WEIGHT_CUT
    keep_f = e < e[keep_i].max() + 30 * OMEGA_E  # final states up to 30 quanta above
    xnm = u[:, keep_i].T @ (grid[:, None] * u[:, keep_f])
    z = np.exp(-beta * e[keep_i]).sum()
    w, de = kubo_weights(beta, e[keep_i], e[keep_f], z)
    amp = 3.0 * w * xnm**2 * mirror(keep_i[keep_f])
    vn = (u[:, keep_i] ** 2 * potential(grid)[:, None]).sum(0)
    pops = np.exp(-beta * e[keep_i]) / z
    statics = {"V_per_particle": 3.0 * float(pops @ vn),
               "E_per_particle": 3.0 * float(pops @ (e[keep_i] + 0.0)) + 3.0 * float(
                   sinc_dvr(grid, potential(grid), MU)[0][0])}
    return de.ravel(), amp.ravel(), statics


def lines_s2(beta, grid, l_max=None):
    """Radial Morse lines over l → l ± 1 (frequency, C_xx amplitude for the vector q) and statics."""
    states = []  # (l, energies, radial vectors)
    l = 0
    e_ground = None
    while True:
        v = morse(grid) + l * (l + 1) / (2.0 * MU * grid**2)
        e, u = sinc_dvr(grid, v, MU)
        if e_ground is None:
            e_ground = e[0]
        states.append((l, e - e_ground, u, v))
        if (l_max is not None and l >= l_max) or (
                l_max is None and np.exp(-beta * (e[0] - e_ground)) < WEIGHT_CUT and l > 2):
            break
        l += 1
    z = sum((2 * l + 1) * np.exp(-beta * e).sum() for l, e, _, _ in states)
    freqs, amps = [], []
    v_mean = 0.0
    for idx, (l, e, u, v) in enumerate(states):
        keep_i = np.exp(-beta * e) > WEIGHT_CUT
        if not keep_i.any():
            continue
        pops = (2 * l + 1) * np.exp(-beta * e[keep_i]) / z
        v_mean += float(pops @ (u[:, keep_i] ** 2 * morse(grid)[:, None]).sum(0))
        for lf in (l - 1, l + 1):
            if lf < 0 or lf >= len(states):
                continue
            _, ef, uf, _ = states[lf]
            keep_f = ef < e[keep_i].max() + 30 * OMEGA_E
            r = u[:, keep_i].T @ (grid[:, None] * uf[:, keep_f])
            # Σ_{M, M', c} |<n l M| q_c |n' l' M'>|² = max(l, l') |R|²; the (2l+1) of the initial
            # level is inside this sum, so the Boltzmann factor carries no extra degeneracy.
            w, de = kubo_weights(beta, e[keep_i], ef[keep_f], z)
            kept_f = (np.exp(-beta * ef) > WEIGHT_CUT)[keep_f]
            amps.append((max(l, lf) * w * r**2 * mirror(kept_f)).ravel())
            freqs.append(de.ravel())
    e_mean = sum(float(((2 * l + 1) * np.exp(-beta * e) * (e + e_ground)).sum())
                 for l, e, _, _ in states) / z
    statics = {"V_per_particle": v_mean, "E_per_particle": e_mean, "l_max": len(states) - 1}
    return np.concatenate(freqs), np.concatenate(amps), statics


def grid_for(system, points):
    lo, hi = (R_E - 1.0, R_E + 4.0)
    return np.linspace(lo, hi, points)


def reference(system, t_kelvin, times, points=600):
    """Exact C_xx(t), C_vv(t) (Y-4 normalisation) on `times` (ħ/Eh) and static values."""
    beta = 1.0 / (K_B * t_kelvin)
    grid = grid_for(system, points)
    f, amp, statics = (lines_s1 if system == "s1" else lines_s2)(beta, grid)
    times = np.asarray(times, dtype=float)
    cos = np.cos(np.outer(times, f))
    cxx = cos @ amp
    cvv = cos @ (amp * f**2)
    return {"t": times, "c_xx": cxx, "c_vv": cvv, "lines_f": f, "lines_a": amp, **statics,
            "c_vv0_exact": 3.0 / (beta * MU)}


def check():
    ok = True
    print(f"mu = {MU!r} m_e = {MU / AMU!r} amu; omega_e = {OMEGA_E * 219474.6313632:.2f} cm^-1")
    t = np.linspace(0.0, 2000.0, 9)
    for system in ("s1", "s2"):
        for tk in (300.0, 100.0):
            r1 = reference(system, tk, t, points=600)
            r2 = reference(system, tk, t, points=900)
            dx = np.abs(r1["c_xx"] - r2["c_xx"]).max() / np.abs(r2["c_xx"]).max()
            dv = np.abs(r1["c_vv"] - r2["c_vv"]).max() / np.abs(r2["c_vv"]).max()
            sum_rule = r2["c_vv"][0] / r2["c_vv0_exact"] - 1.0
            print(f"{system} {tk:5.0f} K: grid 600 vs 900: C_xx {dx:.1e}, C_vv {dv:.1e}; "
                  f"C_vv(0)/(3/(βμ)) - 1 = {sum_rule:.1e}; <V> = {r2['V_per_particle']:.8e} Eh; "
                  f"C_xx(0) = {r2['c_xx'][0]:.8e}"
                  + (f"; l_max {r2['l_max']}" if system == "s2" else ""))
            ok &= dx < 1e-8 and dv < 1e-8 and abs(sum_rule) < 1e-8
    # Harmonic limit: V = ½ μ ω² (x - r_e)², Kubo C_xx = 3 [cos(ωt)/(βμω²) + r_e²] (s1 normalisation)
    beta = 1.0 / (K_B * 300.0)
    grid = grid_for("s1", 600)
    f, amp, _ = lines_s1(beta, grid, potential=lambda x: 0.5 * MU * OMEGA_E**2 * (x - R_E) ** 2)
    cxx = np.cos(np.outer(t, f)) @ amp
    ana = 3.0 * (np.cos(OMEGA_E * t) / (beta * MU * OMEGA_E**2) + R_E**2)
    dh = np.abs(cxx - ana).max() / ana.max()
    print(f"harmonic limit (s1, 300 K): max |C_xx - analytic| / max = {dh:.1e}")
    ok &= dh < 1e-8
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(check() if a.check else 0)
