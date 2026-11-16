"""M11 tier 2 reference: J = 0 vibrational levels of H₂O on a MACE surface (3D DVR, Radau coordinates).

    pixi run -e ml python studies/m11-dynamics-failure/water_ref.py MODEL [--device cuda]
        [--n-r 32] [--n-theta 40] [--out results/tier2/reference.json] [--check]

Atomic units. Radau coordinates for H₂O (Smith 1980): with G the midpoint of the two H atoms and
d = G − x_O, the canonical point is P = G − α d, α = (m_O/M)^{1/2}, and the Radau vectors are
r_i = x_Hi − P. Then 2T = m_H (|ṙ1|² + |ṙ2|²) + M|Ẋ|², so for J = 0 and u = r1 r2 ψ:
    H = −1/(2m_H) (∂²/∂r1² + ∂²/∂r2²) + (1/(2m_H r1²) + 1/(2m_H r2²)) L̂² + V(r1, r2, θ),
L̂² = −(1/sin θ) ∂_θ sin θ ∂_θ. DVR: Colbert–Miller sinc in r1, r2; Gauss–Legendre in cos θ
(L̂² diagonal, l(l+1), in the Legendre basis). Lowest states by Lanczos (scipy eigsh).

Also: the minimum (BFGS), the Cartesian harmonic frequencies and normal modes (finite-difference
Hessian of the client's forces), and the Y-4 weight of each mode in the atomic C_vv,
w_k ∝ Σ_i L_ik²/m_i (L mass-weighted, unit norm).
Checks (--check):
  (a) Wilson GF frequencies in Radau coordinates (G diagonal: 1/m_H, 1/m_H, 1/(m_H r1²) + 1/(m_H r2²))
      equal the Cartesian harmonic frequencies (tests the Radau construction and masses);
  (b) the DVR on the Radau-quadratic surface scaled by λ = 400 gives (E_n − E_0)/√λ → the GF
      frequencies (tests the kinetic energy operator);
  (c) grid convergence of the MACE levels (n-r, n-theta increased by 8 each).
"""

import argparse
import json
import os
import sys

import numpy as np
from numpy.polynomial import legendre
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import water_client as wc  # noqa: E402

AMU = 1822.888486209  # m_e, U-2
CM = 219474.6313632  # cm⁻¹ per Eh, U-2
M_O, M_H = 15.999 * AMU, 1.008 * AMU
MASSES = np.array([M_O, M_H, M_H])
M_TOT = MASSES.sum()
ALPHA = np.sqrt(M_O / M_TOT)
CAP_EH = 0.5  # ≈ 110 000 cm⁻¹ above the minimum
HH_CAP_BOHR = 1.2


class Surface:
    """Per-molecule MACE energies for many geometries, batched (molecules do not interact)."""

    def __init__(self, model, device, batch=2048):
        self.model, self.device, self.batch = model, device, batch
        self.client, _ = wc.client_for(model, batch, device)
        self.one, _ = wc.client_for(model, 1, device)

    def energies(self, geoms):
        """geoms `[n, 3, 3]` (bohr; O, H, H) -> energies `[n]` (Eh), sum of MACE node energies."""
        torch, c = self.client.torch, self.client
        out = np.empty(len(geoms))
        for s in range(0, len(geoms), self.batch):
            g = geoms[s:s + self.batch]
            n = len(g)
            pad = np.concatenate([g, np.repeat(g[-1:], self.batch - n, 0)]) if n < self.batch else g
            q = torch.tensor(pad.reshape(-1, 3), dtype=torch.float64, device=c.device)
            v = q[c.cj] - q[c.ci]
            inside = (v * v).sum(-1) < c.r_max * c.r_max
            i, j = c.ci[inside], c.cj[inside]
            zeros = torch.zeros(i.numel(), 3, dtype=c.dtype, device=c.device)
            data = {"positions": (q * wc.mc.BOHR_IN_ANGSTROM).to(c.dtype), "node_attrs": c.node_attrs,
                    "edge_index": torch.stack([i, j]), "shifts": zeros, "unit_shifts": zeros,
                    "cell": c.cell, "batch": c.batch, "ptr": c.ptr}
            with torch.no_grad():
                o = c.model(data, training=False, compute_force=False, compute_virials=False)
            e = o["node_energy"].detach().to(torch.float64).reshape(-1, 3).sum(1).cpu().numpy()
            out[s:s + n] = e[:n] / wc.mc.HARTREE_IN_EV
        return out

    def energy_forces(self, geom):
        return self.one.energy_forces(geom)


def radau_to_cart(r1, r2, th):
    """Radau (r1, r2, θ) arrays -> Cartesian `[n, 3, 3]` (O, H1, H2), COM at the origin, in-plane."""
    r1, r2, th = np.broadcast_arrays(np.asarray(r1, float), np.asarray(r2, float), np.asarray(th, float))
    v1 = np.stack([r1, np.zeros_like(r1), np.zeros_like(r1)], -1)
    v2 = np.stack([r2 * np.cos(th), r2 * np.sin(th), np.zeros_like(r2)], -1)
    d = (v1 + v2) / (2.0 * ALPHA)
    s = v1 - v2
    g = (M_O / M_TOT) * d  # COM at 0: G = (M_O/M) d
    return np.stack([g - d, g + s / 2.0, g - s / 2.0], -2)


def cart_to_radau(x):
    o, h1, h2 = x
    g = (h1 + h2) / 2.0
    p = g - ALPHA * (g - o)
    v1, v2 = h1 - p, h2 - p
    r1, r2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return r1, r2, np.arccos(np.dot(v1, v2) / (r1 * r2))


def minimum(surf):
    r, th = 0.9572 / wc.mc.BOHR_IN_ANGSTROM, np.radians(104.52)
    x0 = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0], [r * np.cos(th), r * np.sin(th), 0.0]])

    def f(x):
        e, frc = surf.energy_forces(x.reshape(3, 3))
        return e, -frc.ravel()

    res = minimize(f, x0.ravel(), jac=True, method="BFGS", options={"gtol": 1e-9, "maxiter": 500})
    x = res.x.reshape(3, 3)
    return x - (MASSES[:, None] * x).sum(0) / M_TOT, float(res.fun), float(np.abs(res.jac).max())


def harmonic(surf, x, h=1e-3):
    """Cartesian harmonic frequencies (cm⁻¹, the three largest), mass-weighted modes, Y-4 weights."""
    hess = np.zeros((9, 9))
    for k in range(9):
        p, q = x.ravel().copy(), x.ravel().copy()
        p[k] += h
        q[k] -= h
        hess[:, k] = -(surf.energy_forces(p.reshape(3, 3))[1].ravel()
                       - surf.energy_forces(q.reshape(3, 3))[1].ravel()) / (2 * h)
    hess = 0.5 * (hess + hess.T)
    m = np.repeat(MASSES, 3)
    w2, L = np.linalg.eigh(hess / np.sqrt(np.outer(m, m)))
    idx = np.argsort(w2)[-3:]  # bend, sym, antisym stretch (ascending)
    freqs = np.sqrt(w2[idx]) * CM
    modes = L[:, idx]
    weights = (modes**2 / m[:, None]).sum(0)
    return freqs, modes, weights / weights.sum()


def gf_frequencies(surf, rad, h=(2e-3, 2e-3, 2e-3)):
    """Wilson GF in Radau coordinates: F by finite differences of the MACE energy."""
    rad = np.array(rad)
    pts, idx = [], {}
    for a in range(3):
        for b in range(3):
            for sa in (-1, 1):
                for sb in (-1, 1):
                    d = np.zeros(3)
                    d[a] += sa * h[a]
                    d[b] += sb * h[b]
                    idx[a, b, sa, sb] = len(pts)
                    pts.append(rad + d)
    pts = np.array(pts)
    e = surf.energies(radau_to_cart(pts[:, 0], pts[:, 1], pts[:, 2]))
    F = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            F[a, b] = (e[idx[a, b, 1, 1]] - e[idx[a, b, 1, -1]] - e[idx[a, b, -1, 1]]
                       + e[idx[a, b, -1, -1]]) / (4 * h[a] * h[b])
    G = np.diag([1 / M_H, 1 / M_H, 1 / (M_H * rad[0] ** 2) + 1 / (M_H * rad[1] ** 2)])
    w2 = np.linalg.eigvals(G @ F).real
    return np.sort(np.sqrt(w2)) * CM, F


def sinc_kinetic(x, mass):
    n, dx = len(x), x[1] - x[0]
    d = np.arange(n)[:, None] - np.arange(n)[None, :]
    with np.errstate(divide="ignore"):
        t = np.where(d == 0, np.pi**2 / 3.0, 2.0 / np.where(d == 0, 1, d) ** 2)
    return t * (-1.0) ** np.abs(d) / (2.0 * mass * dx * dx)


def grid(rad_e, n_r, n_th, scale=1.0, lo=0.55, hi=1.1):
    r = np.linspace(rad_e[0] - lo * scale, rad_e[0] + hi * scale, n_r)
    x, wts = legendre.leggauss(n_th)
    th = np.arccos(x)
    # Normalised Legendre functions at the points: T[k, l] = sqrt(w_k) sqrt(l + 1/2) P_l(x_k)
    T = np.array([np.sqrt(wts) * np.sqrt(l + 0.5) * legendre.legval(x, np.eye(n_th)[l])
                  for l in range(n_th)]).T
    L2 = T @ np.diag([l * (l + 1.0) for l in range(n_th)]) @ T.T
    return r, th, L2


def solve(r, th, L2, v, k=12):
    """Lowest k levels (Eh) of the Radau J = 0 Hamiltonian on the grid; v `[n_r, n_r, n_th]`."""
    n_r, n_th = len(r), len(th)
    K = sinc_kinetic(r, M_H)
    c = 1.0 / (2 * M_H * r[:, None] ** 2) + 1.0 / (2 * M_H * r[None, :] ** 2)
    shape = (n_r, n_r, n_th)

    def mv(x):
        p = x.reshape(shape)
        out = np.einsum("ab,bjk->ajk", K, p) + np.einsum("ab,ibk->iak", K, p)
        out += c[:, :, None] * (p @ L2.T) + v * p
        return out.ravel()

    op = LinearOperator((np.prod(shape),) * 2, matvec=mv, dtype=float)
    # Shift-free Lanczos for the lowest states; v0 fixed for reproducibility.
    e, vec = eigsh(op, k=k, which="SA", v0=np.ones(np.prod(shape)), tol=1e-11, maxiter=200000)
    order = np.argsort(e)
    return e[order], vec[:, order].T.reshape((k,) + shape)


def assign(e, vecs, rad_e, r, th):
    """ν2, ν1, ν3, 2ν2 from the lowest levels: parity under r1 ↔ r2 and the θ / r character."""
    out = []
    for n, (en, v) in enumerate(zip(e, vecs)):
        par = float(np.sum(v * np.transpose(v, (1, 0, 2))) / np.sum(v * v))
        p = v**2 / np.sum(v**2)
        dth = float(np.sqrt(np.sum(p * (th[None, None, :] - rad_e[2]) ** 2)))
        drs = float(np.sqrt(np.sum(p * ((r[:, None, None] - rad_e[0]) ** 2 + (r[None, :, None] - rad_e[1]) ** 2))))
        out.append({"n": n, "E_cm": float((en - e[0]) * CM), "parity": round(par, 3),
                    "rms_dtheta": dth, "rms_dr": drs})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-r", type=int, default=32)
    ap.add_argument("--n-theta", type=int, default=40)
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    surf = Surface(a.model, a.device)
    x, e_min, gmax = minimum(surf)
    rad_e = cart_to_radau(x)
    print(f"minimum: E = {e_min:.10f} Eh, max |grad| = {gmax:.1e}; Radau r1 = {rad_e[0]:.5f}, "
          f"r2 = {rad_e[1]:.5f} bohr, θ = {np.degrees(rad_e[2]):.3f}°; "
          f"OH = {np.linalg.norm(x[1] - x[0]):.5f} bohr, HOH = "
          f"{np.degrees(np.arccos(np.dot(x[1] - x[0], x[2] - x[0]) / np.linalg.norm(x[1] - x[0]) / np.linalg.norm(x[2] - x[0]))):.3f}°")
    freqs, modes, weights = harmonic(surf, x)
    gf, F = gf_frequencies(surf, rad_e)
    print(f"harmonic (Cartesian): {np.round(freqs, 2)} cm⁻¹; GF (Radau): {np.round(gf, 2)}; "
          f"Y-4 weights {np.round(weights, 4)}")
    ok = bool(np.abs(freqs - gf).max() < 0.5)

    def levels(n_r, n_th, quad_lambda=None, scale=1.0, lo=0.55, hi=1.1):
        r, th, L2 = grid(rad_e, n_r, n_th, scale, lo, hi)
        R1, R2, TH = np.meshgrid(r, r, th, indexing="ij")
        if quad_lambda is None:
            xs = radau_to_cart(R1.ravel(), R2.ravel(), TH.ravel())
            v = surf.energies(xs).reshape(R1.shape) - e_min
            # The surface has holes (hundreds of Eh below the minimum) at H–H < 1 bohr, where the
            # physical surface is > 20 000 cm⁻¹ up: set V to the cap there and above the cap.
            capped = (np.linalg.norm(xs[:, 1] - xs[:, 2], axis=1) < HH_CAP_BOHR).reshape(R1.shape)
            v = np.where(capped, CAP_EH, np.minimum(v, CAP_EH))
            if v.min() < -1e-8:
                sys.exit(f"water_ref: the surface has a hole on the grid ({v.min():.3e} Eh below the minimum)")
        else:
            g = np.stack([R1 - rad_e[0], R2 - rad_e[1], TH - rad_e[2]], -1)
            v = quad_lambda * 0.5 * np.einsum("...a,ab,...b->...", g, F, g)
        e, vec = solve(r, th, L2, v)
        if quad_lambda is None:
            w = (vec**2 * (v >= CAP_EH)).sum(axis=(1, 2, 3)) / (vec**2).sum(axis=(1, 2, 3))
            print(f"grid {n_r}/{n_th}: {int((v >= CAP_EH).sum())} capped points; max weight of the lowest "
                  f"{len(e)} states on capped points {w.max():.1e}")
            # A weight w on the cap shifts a level by at most w × CAP_EH (1e-6 → 0.1 cm⁻¹).
            if w.max() > 1e-6:
                sys.exit("water_ref: the states reach the capped region")
        return e, vec, r, th

    if a.check:
        # The quadratic surface scaled by λ narrows the states by λ^(-1/4): the r range shrinks and
        # the Legendre count grows by that factor. The excitation nearest each GF frequency, over
        # √λ, differs from it by δ(λ) = δ∞ + c1/√λ + c2/λ + … (the curvilinear kinetic energy and
        # anharmonic couplings it brings); δ∞, from a fit over three λ, must vanish.
        dev = {}
        lams = (16.0, 64.0, 256.0)
        for lam in lams:
            sc = lam ** -0.25
            # symmetric r range (±1.1 bohr before scaling, 6σ of the scaled ground state)
            e, *_ = levels(a.n_r + 8, int(round(a.n_theta / sc)), lam, sc, 1.1, 1.1)
            exc = (e[1:] - e[0]) * CM / np.sqrt(lam)
            dev[lam] = np.array([exc[np.argmin(np.abs(exc - g))] - g for g in gf])
            print(f"(b) quadratic surface × {lam:g}: δ = {np.round(dev[lam], 3)} cm⁻¹")
        A_fit = np.array([[1.0, lam ** -0.5, 1.0 / lam] for lam in lams])
        d_inf = np.linalg.solve(A_fit, np.array([dev[lam] for lam in lams]))[0]
        print(f"(b) extrapolated δ∞ = {np.round(d_inf, 3)} cm⁻¹ (GF {np.round(gf, 2)})")
        ok &= bool(np.abs(d_inf).max() < 0.5)

    e, vec, r, th = levels(a.n_r, a.n_theta)
    tab = assign(e, vec, rad_e, r, th)
    zpe = float(e[0] * CM)
    print(f"ZPE = {zpe:.2f} cm⁻¹ = {e[0]:.8f} Eh")
    for t in tab[:8]:
        print(f"  n={t['n']} {t['E_cm']:9.2f} cm⁻¹ parity {t['parity']:+.3f} rms dθ {t['rms_dtheta']:.3f} rms dr {t['rms_dr']:.3f}")
    if a.check:
        for label, kw in ((f"{a.n_r + 8}/{a.n_theta + 8}", dict(n_r=a.n_r + 8, n_th=a.n_theta + 8)),
                          ("r range −0.75/+1.4 bohr", dict(n_r=a.n_r + 8, n_th=a.n_theta, lo=0.75, hi=1.4))):
            e2, *_ = levels(**kw)
            d = np.abs((e2[:8] - e2[0]) - (e[:8] - e[0])).max() * CM
            dz = abs(e2[0] - e[0]) * CM
            print(f"(c) grid {a.n_r}/{a.n_theta} vs {label}: max level change {d:.3f} cm⁻¹, ZPE change {dz:.3f} cm⁻¹")
            ok &= d < 0.1 and dz < 0.1
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"model": os.path.basename(a.model), "grid": [a.n_r, a.n_theta],
                       "minimum_bohr": x.tolist(), "E_min_eh": e_min, "radau_e": list(rad_e),
                       "harmonic_cm": freqs.tolist(), "gf_cm": gf.tolist(), "y4_weights": weights.tolist(),
                       "zpe_eh": float(e[0]), "levels": tab}, f, indent=1)
    if a.check:
        print("checks", "passed" if ok else "FAILED")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
