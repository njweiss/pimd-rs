"""Minimum, normal modes and exact P-bead harmonic energies of one water molecule (M6c plan,
"Protocol" step 3; docs/experiments/m6-water-mace.md).

    pixi run -e ml  python studies/m6-water-mace/harmonic.py mace
    pixi run -e ref python studies/m6-water-mace/harmonic.py xtb

Energies and forces come from the same calculators as the study's force evaluations: MACE through
mace's `MACECalculator` on the MACE-OFF23 small model file (float64, CPU), GFN2-xTB through
`tblite.interface.Calculator` (as `ipi_client.py xtb`). Atomic units throughout (bohr, Eh, m_e,
ħ = 1); constants CODATA 2018 as SPEC U-2, written out because this script is outside the engine.

Minimum: Newton steps with the finite-difference Hessian (pseudo-inverse without the six external
modes) until max |force| < 1e-6 Eh/bohr. Hessian: central differences of the forces, step h =
1e-3 bohr, checked against h = 5e-4 bohr (largest frequency change reported). Normal modes of the
mass-weighted Hessian for H2O, D2O, HDO.

Exact P-bead harmonic energy of one mode at inverse temperature β (the energy both PIMD estimators
estimate for a harmonic potential): with ω_k = 2 (P/β) sin(πk/P),
    Z_P = Π_k P / (β sqrt(ω_k² + ω²)),   E_P = -∂ ln Z_P/∂β = Σ_k ω² / (β (ω_k² + ω²)),
which gives 1/β at P = 1 and (ω/2) coth(βω/2) as P → ∞ (both checked below). The molecule's
harmonic energy above V_min is 3/β (translation and rotation, kinetic only) + Σ_modes E_P.
Output: studies/m6-water-mace/results/harmonic_<model>.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

BOHR_IN_ANGSTROM = 0.529177210903
HARTREE_IN_EV = 27.211386245988
AMU_IN_ME = 1822.888486209
K_B = 3.1668115634556e-6  # Eh/K
HARTREE_IN_CM = 219474.6313632
T = 300.0
PS = [1, 2, 4, 8, 16, 32, 64, 128]
MASSES = {"h2o": [15.999, 1.008, 1.008], "d2o": [15.999, 2.014, 2.014], "hdo": [15.999, 2.014, 1.008]}
SYMBOLS = ["O", "H", "H"]
START = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]) / BOHR_IN_ANGSTROM
ROOT = Path(__file__).resolve().parents[2]
MACE_MODEL = ROOT / "ff_models/mace_off23/MACE-OFF23_small.model"


def mace_backend():
    import torch
    from ase import Atoms
    from mace.calculators import MACECalculator

    torch.set_num_threads(1)
    calc = MACECalculator(model_paths=str(MACE_MODEL), device="cpu", default_dtype="float64")

    def ef(x):
        atoms = Atoms(SYMBOLS, positions=x * BOHR_IN_ANGSTROM, pbc=False)
        atoms.calc = calc
        e = atoms.get_potential_energy() / HARTREE_IN_EV
        return e, atoms.get_forces() * (BOHR_IN_ANGSTROM / HARTREE_IN_EV)

    return ef, {"model": str(MACE_MODEL.relative_to(ROOT)), "calculator": "mace MACECalculator float64 cpu"}


def xtb_backend():
    from tblite.interface import Calculator

    numbers = np.array([8, 1, 1])

    def ef(x):
        calc = Calculator("GFN2-xTB", numbers, x)
        calc.set("verbosity", 0)
        res = calc.singlepoint()
        return float(res.get("energy")), -np.asarray(res.get("gradient"))

    return ef, {"model": "GFN2-xTB", "calculator": "tblite.interface.Calculator"}


def hessian(ef, x, h):
    """Central-difference Hessian, 9×9, Eh/bohr², symmetrised."""
    n = x.size
    hess = np.zeros((n, n))
    for i in range(n):
        d = np.zeros(n)
        d[i] = h
        fp = ef((x.ravel() + d).reshape(x.shape))[1].ravel()
        fm = ef((x.ravel() - d).reshape(x.shape))[1].ravel()
        hess[i] = -(fp - fm) / (2 * h)
    return 0.5 * (hess + hess.T)


def minimise(ef, x):
    for it in range(50):
        e, f = ef(x)
        if np.abs(f).max() < 1e-6:
            return x, e, f, it
        w, v = np.linalg.eigh(hessian(ef, x, 1e-3))
        keep = np.abs(w) > 1e-4
        step = v[:, keep] @ ((v[:, keep].T @ f.ravel()) / w[keep])
        x = x + step.reshape(x.shape)
    sys.exit("minimise: no convergence in 50 Newton steps")


def modes(hess, masses_amu):
    m = np.repeat(np.array(masses_amu) * AMU_IN_ME, 3)
    w = np.linalg.eigvalsh(hess / np.sqrt(np.outer(m, m)))
    ext, vib = w[:6], w[6:]
    return np.sqrt(vib), ext


def e_p(omega, beta, p):
    k = np.arange(p)
    wk2 = (2 * p / beta * np.sin(np.pi * k / p)) ** 2
    return float(np.sum(omega**2 / (beta * (wk2 + omega**2))))


def main():
    which = sys.argv[1]
    ef, meta = {"mace": mace_backend, "xtb": xtb_backend}[which]()
    beta = 1.0 / (K_B * T)
    x, vmin, f, iters = minimise(ef, START.copy())
    r = [np.linalg.norm(x[i] - x[0]) for i in (1, 2)]
    a, b = x[1] - x[0], x[2] - x[0]
    theta = np.degrees(np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    h1, h2 = hessian(ef, x, 1e-3), hessian(ef, x, 5e-4)

    # Self-checks of the harmonic formula.
    w0 = 0.017
    assert abs(e_p(w0, beta, 1) - 1 / beta) < 1e-15
    exact = 0.5 * w0 / np.tanh(0.5 * beta * w0)
    assert abs(e_p(w0, beta, 65536) - exact) < 1e-7 * exact  # error falls as 1/P²: 2.4e-6 at 4096

    out = {"model": meta, "temperature_k": T, "v_min_eh": vmin, "max_force_eh_bohr": float(np.abs(f).max()),
           "newton_steps": iters, "geometry": {"positions_bohr": x.tolist(), "r_oh_bohr": r,
           "theta_hoh_deg": float(theta)}, "isotopes": {}}
    for sp, masses in MASSES.items():
        om, ext = modes(h1, masses)
        om2, _ = modes(h2, masses)
        e = {str(p): 3 / beta + sum(e_p(w, beta, p) for w in om) for p in PS}
        e["inf"] = 3 / beta + sum(0.5 * w / np.tanh(0.5 * beta * w) for w in om)
        out["isotopes"][sp] = {
            "masses_amu": masses,
            "frequencies_cm": (om * HARTREE_IN_CM).tolist(),
            "frequency_change_h_vs_h2_cm": float(np.abs(om - om2).max() * HARTREE_IN_CM),
            "external_eigenvalues_au": ext.tolist(),
            "zpe_eh": float(0.5 * om.sum()),
            "e_minus_vmin_eh": e,
        }
    path = ROOT / f"studies/m6-water-mace/results/harmonic_{which}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("v_min_eh", "geometry")}, indent=1))
    for sp, d in out["isotopes"].items():
        print(sp, ["%.1f" % w for w in d["frequencies_cm"]], "ΔE(P)", {p: round(v, 6) for p, v in d["e_minus_vmin_eh"].items()},
              "fd check %.2g cm-1" % d["frequency_change_h_vs_h2_cm"])


if __name__ == "__main__":
    main()
