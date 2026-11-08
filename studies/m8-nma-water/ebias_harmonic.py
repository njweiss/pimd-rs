"""BAOAB step-size bias of ⟨E_prim⟩ − ⟨E_vir⟩ for harmonic physical modes (docs/experiments/
m8-nma-water.md, log 2026-11-08). Exact stationary position variance ⟨Q_k²⟩ per ring-polymer
normal mode of B A O A B + PILE-L (as tbias_harmonic.py), put into the primitive and centroid-
virial estimators (E-1, E-2) and compared with the exact ⟨Q_k²⟩ = kT_P/(ω_k² + ω²); same water
mode model, per atom = per water / 3, 300 K, τ0 = 100 fs, Δt = 0.5 and 0.25 fs.

    pixi run -e ref python studies/m8-nma-water/ebias_harmonic.py
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
c = 2*np.pi*2.99792458e-5
KB = 3.166811563e-6  # Eh/K
def qq(w, wk, gam, dt):
    B = np.array([[1, 0], [-w*w*dt/2, 1]])
    if wk == 0: A = np.array([[1, dt/2], [0, 1]])
    else:
        a = wk*dt/2; A = np.array([[np.cos(a), np.sin(a)/wk], [-wk*np.sin(a), np.cos(a)]])
    e = np.exp(-gam*dt); O = np.diag([1, e]); Q0 = np.diag([0, 1-e*e])
    M = B@A@O@A@B; N = B@A
    return solve_discrete_lyapunov(M, N@Q0@N.T)[0, 0]   # m = 1, kT_P = 1
def diff(nu, P, dt, tau0=100.0, T=300.0):
    wP = P*T*0.69503476*c; w = nu*c
    ks = range(P); wk = [2*wP*np.sin(np.pi*k/P) for k in ks]
    Q = [qq(w, wk[k], 1/tau0 if k == 0 else 2*wk[k], dt) for k in ks]
    Qx = [1/(wk[k]**2 + w*w) for k in ks]
    def est(Q):
        prim = 0.5 - sum(wk[k]**2/2*Q[k] for k in ks)/P + sum(w*w/2*Q[k] for k in ks)/P
        vir = 0.5/P + sum(w*w/2*Q[k] for k in ks)/P + w*w/(2*P)*sum(Q[k] for k in ks if k)
        return prim, vir                 # units kT_P / P = kT? no: kT_P units, see below
    (p, v), (px, vx) = est(Q), est(Qx)
    return (p - v), (px - vx), p - px, v - vx
modes = [(3700, 2), (1600, 1), (600, 3)]
for P in (1, 8, 16, 32):
    for dt in (0.5, 0.25):
        kTP = P*300*KB
        # est() in units of kT_P with kT = kT_P/P: prim/vir terms per dof
        d = [sum(n*diff(nu, P, dt)[i] for nu, n in modes)/9*3*kTP for i in range(4)]
        print(f"P={P:2d} dt={dt}: per atom Eh: prim-vir {d[0]:+.2e} (exact {d[1]:+.1e}); prim err {d[2]:+.2e}, vir err {d[3]:+.2e}")
