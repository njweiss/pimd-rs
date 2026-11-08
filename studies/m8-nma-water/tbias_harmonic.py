"""BAOAB end-of-step kinetic bias of T_est for harmonic physical modes, the b_P of the amended run
health criterion (docs/experiments/m8-nma-water.md). Exact stationary ⟨p²⟩ of the linear map
B A O A B per ring-polymer normal mode (A = exact free rotation at ω_k = 2 ω_P sin(kπ/P), O =
PILE-L with γ_k = 2 ω_k, centroid 1/τ0), averaged over modes; a water's 9 degrees of freedom
taken as two O–H stretches (3 700 cm⁻¹), one bend (1 600 cm⁻¹), three librations (600 cm⁻¹) and
three translations (no bias). 300 K, τ0 = 100 fs, Δt = 0.5 and 0.25 fs.

    pixi run -e ref python studies/m8-nma-water/tbias_harmonic.py
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
c = 2*np.pi*2.99792458e-5        # cm^-1 -> rad/fs
def ratio(w, wk, gam, dt):
    B = np.array([[1, 0], [-w*w*dt/2, 1]])          # (q, p), m = 1, kT_P = 1
    if wk == 0: A = np.array([[1, dt/2], [0, 1]])
    else:
        a = wk*dt/2; A = np.array([[np.cos(a), np.sin(a)/wk], [-wk*np.sin(a), np.cos(a)]])
    e = np.exp(-gam*dt); O = np.diag([1, e]); Q0 = np.diag([0, 1-e*e])
    M = B@A@O@A@B; N = B@A                         # noise enters after O, then A, B
    Sig = solve_discrete_lyapunov(M, N@Q0@N.T)
    return Sig[1, 1]
def bias(nu, P, dt=0.5, tau0=100.0, T=300.0):
    wP = P*T*0.69503476*c                          # P kT/hbar in rad/fs
    w = nu*c; r = []
    for k in range(P):
        wk = 2*wP*np.sin(np.pi*k/P); gam = 1/tau0 if k == 0 else 2*wk
        r.append(ratio(w, wk, gam, dt))
    return np.mean(r) - 1
modes = [(3700, 2), (1600, 1), (600, 3)]           # per water, out of 9 dof (3 translations ~0)
for P in (1, 8, 16, 32):
    for dt in (0.5, 0.25):
        d = sum(n*bias(nu, P, dt) for nu, n in modes)/9
        print(f"P={P:2d} dt={dt}: per-water relative bias {d:+.5f} -> {300*d:+.2f} K;", " ".join(f"{nu}:{bias(nu,P,dt):+.4f}" for nu,_ in modes))
