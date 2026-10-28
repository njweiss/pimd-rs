"""Reference data for the double-well PIMD test (MILESTONE_2, SPEC P-2, E-4, E-9).

Run from the repository root:

    pixi run -e ref python crates/pimd-estimators/tests/data/gen_double_well.py [--out DIR]

Writes double_well.json next to this script (or into DIR). Everything is in Hartree atomic units
(SPEC U-1, hbar = 1). The system is one particle in 1D with V(x) = a x^4 - b x^2; the 3D separable
`DoubleWell` of SPEC P-2 is three independent copies, so the 3D energy of N atoms is 3 N times the
1D value stored here.

Two quantities, each with a convergence table:

1. exact_energy_1d: the quantum thermal energy <H> = sum_n E_n exp(-beta E_n) / Z, from exact
   diagonalisation of H = p^2/(2m) + V on a uniform Colbert-Miller sinc-DVR grid on [-L, L]
   (D. T. Colbert, W. H. Miller, J. Chem. Phys. 96, 1982 (1992)), converged in L and grid size.

2. finite_p_energy_1d[P]: the exact expectation of the primitive/virial energy estimators of the
   P-bead ring polymer sampled by SPEC RP-3 (distribution exp(-(beta/P) H_P), omega_P = P/beta),
   i.e. -d ln Z_P / d beta with
       Z_P = Tr K^P,  K(x, y) = sqrt(m P / (2 pi beta)) exp(-m P (x - y)^2 / (2 beta)
                                 - (beta / (2 P)) (V(x) + V(y))) dx,
   evaluated on a uniform grid (transfer matrix). The beta-derivative is analytic:
       -d ln Z_P / d beta = -P Tr(K^(P-1) dK/dbeta) / Tr(K^P),
       dK/dbeta = K * (-1/(2 beta) + m P (x - y)^2 / (2 beta^2) - (V(x) + V(y)) / (2 P)).
   This is the finite-P analogue of SPEC E-9 for the double well. It is checked here against
   the closed-form E-9 harmonic result (1D: (1/beta) sum_k omega^2 / (omega_k^2 + omega^2)) and the
   DVR against (omega/2) coth(beta omega / 2), so both methods are validated on a solvable case.

Parameters (documented choice, MILESTONE_2 "DoubleWell"): m = 1, a = 1/8, b = 1, beta = 2.
Minima at x = +-sqrt(b / (2a)) = +-2, barrier height b^2 / (4a) = 2 Eh, so beta * barrier = 4;
well curvature V'' = 4b gives omega_well = sqrt(4b / m) = 2, so beta * omega_well = 4 (quantum).
The ground state lies 0.90 Eh above the well bottom, below the barrier top (tunnelling regime).
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg

MASS = 1.0
A = 0.125
B = 1.0
BETA = 2.0
P_VALUES = [32, 64]


def double_well(x):
    return A * x**4 - B * x**2


def dvr_thermal_energy(v, mass, beta, half_width, n):
    """<H> at beta on a sinc-DVR grid of n points on [-half_width, half_width]."""
    x = np.linspace(-half_width, half_width, n)
    dx = x[1] - x[0]
    d = np.arange(n)[:, None] - np.arange(n)[None, :]
    safe = np.where(d == 0, 1, d)
    t = np.where(d == 0, np.pi**2 / 3.0, 2.0 / safe**2) * (-1.0) ** d / (2.0 * mass * dx**2)
    e = scipy.linalg.eigvalsh(t + np.diag(v(x)))
    w = np.exp(-beta * (e - e[0]))
    return float(np.sum(e * w) / np.sum(w)), e


def transfer_matrix_energy(v, mass, beta, p, half_width, n):
    """-d ln Z_P / d beta for the P-bead primitive ring polymer, on an n-point grid."""
    x = np.linspace(-half_width, half_width, n)
    dx = x[1] - x[0]
    vx = v(x)
    d2 = (x[:, None] - x[None, :]) ** 2
    vs = vx[:, None] + vx[None, :]
    k = np.sqrt(mass * p / (2.0 * np.pi * beta)) * dx * np.exp(
        -mass * p * d2 / (2.0 * beta) - beta * vs / (2.0 * p)
    )
    dk = k * (-1.0 / (2.0 * beta) + mass * p * d2 / (2.0 * beta**2) - vs / (2.0 * p))
    lam, u = scipy.linalg.eigh(k)
    scale = lam.max()
    r = lam / scale
    # Tr(K^(P-1) dK) = sum_n lam_n^(P-1) (U^T dK U)_nn; eigenvalues scaled by the largest.
    diag = np.einsum("in,ij,jn->n", u, dk, u)
    num = np.sum(r ** (p - 1) * diag) / scale
    den = np.sum(r**p)
    return float(-p * num / den)


def harmonic_checks():
    """Validate both methods on a 1D harmonic oscillator against closed forms (SPEC E-9)."""
    mass, omega, beta = 1.0, 1.0, 4.0

    def v(x):
        return 0.5 * mass * omega**2 * x**2

    out = {"mass": mass, "omega": omega, "beta": beta}
    exact = 0.5 * omega / np.tanh(0.5 * beta * omega)
    dvr, _ = dvr_thermal_energy(v, mass, beta, 10.0, 300)
    out["dvr"] = dvr
    out["coth_formula"] = float(exact)
    assert abs(dvr - exact) < 1e-10, (dvr, exact)
    rows = []
    for p in [4, 16]:
        wp = p / beta
        wk = np.array([2.0 * wp * np.sin(np.pi * k / p) for k in range(p)])
        # Eigenvalues of the cyclic spring matrix: 2 wp sin(pi k / p), k = 0..P-1 (SPEC RP-4/RP-5).
        e9 = float(np.sum(omega**2 / (wk**2 + omega**2)) / beta)
        tm = transfer_matrix_energy(v, mass, beta, p, 10.0, 1200)
        assert abs(tm - e9) < 1e-9, (p, tm, e9)
        rows.append({"p": p, "transfer_matrix": tm, "e9_formula": e9})
    out["finite_p"] = rows
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    dvr_grids = [(5.0, 200), (6.0, 300), (7.0, 400)]
    dvr_rows = []
    for half_width, n in dvr_grids:
        e, levels = dvr_thermal_energy(double_well, MASS, BETA, half_width, n)
        dvr_rows.append({"half_width": half_width, "n": n, "energy": e})
    exact = dvr_rows[-1]["energy"]
    dvr_spread = max(abs(r["energy"] - exact) for r in dvr_rows)
    assert dvr_spread < 1e-10, dvr_spread
    _, levels = dvr_thermal_energy(double_well, MASS, BETA, 7.0, 400)

    tm_grids = [(5.0, 800), (6.0, 1200), (6.5, 1600)]
    finite_p = {}
    tm_rows = []
    for p in P_VALUES:
        vals = []
        for half_width, n in tm_grids:
            e = transfer_matrix_energy(double_well, MASS, BETA, p, half_width, n)
            vals.append(e)
            tm_rows.append({"p": p, "half_width": half_width, "n": n, "energy": e})
        spread = max(abs(e - vals[-1]) for e in vals)
        assert spread < 1e-8, (p, spread)
        finite_p[str(p)] = vals[-1]

    # O(1/P^2) regime check: (E_32 - E_exact) / (E_64 - E_exact) should be close to 4.
    ratio = (finite_p["32"] - exact) / (finite_p["64"] - exact)
    assert 3.8 < ratio < 4.2, ratio

    data = {
        "description": "1D double well V = a x^4 - b x^2, one particle, atomic units; "
        "3D DoubleWell energy of N atoms = 3 N x the 1D values",
        "generator": "gen_double_well.py",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "parameters": {"mass": MASS, "a": A, "b": B, "beta": BETA},
        "derived": {
            "minimum_x": float(np.sqrt(B / (2 * A))),
            "barrier_height": B * B / (4 * A),
            "omega_well": float(np.sqrt(4 * B / MASS)),
            "beta_omega_well": float(BETA * np.sqrt(4 * B / MASS)),
            "beta_barrier": BETA * B * B / (4 * A),
            "lowest_levels_above_minimum": [float(e + B * B / (4 * A)) for e in levels[:4]],
        },
        "exact_energy_1d": exact,
        "finite_p_energy_1d": finite_p,
        "discretization_ratio_32_over_64": float(ratio),
        "convergence": {
            "dvr": dvr_rows,
            "dvr_max_spread": float(dvr_spread),
            "transfer_matrix": tm_rows,
        },
        "harmonic_self_check": harmonic_checks(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "double_well.json"
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
