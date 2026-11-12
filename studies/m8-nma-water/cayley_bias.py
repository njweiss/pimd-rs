"""BAOAB (exact free ring-polymer propagation, I-2) vs BCOCB (Cayley free propagation) step-size
bias for harmonic physical modes (docs/experiments/m8-nma-water.md, writeup proposal 2;
docs/MILESTONE_9.md decision 1). Same linear-map model as ebias_harmonic.py and
tbias_harmonic.py: exact stationary covariance of B X O X B + PILE-L per ring-polymer normal
mode, X = exact rotation or its Cayley approximant, water modes 2 × 3 700, 1 600, 3 × 600 cm⁻¹,
300 K, τ0 = 100 fs, Δt = 0.5 and 0.25 fs. Prints per atom (Eh) E_prim − E_vir, the errors of
E_prim and E_vir against the exact finite-P values, and the kinetic temperature bias.

    pixi run -e ref python studies/m8-nma-water/cayley_bias.py
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
c = 2*np.pi*2.99792458e-5; KB = 3.166811563e-6
def sig(w, wk, gam, dt, cay):
    B = np.array([[1, 0], [-w*w*dt/2, 1]]); h = dt/2
    if wk == 0: A = np.array([[1, h], [0, 1]])
    elif cay:
        x = (wk*h/2)**2; A = np.array([[1-x, h], [-wk*wk*h, 1-x]])/(1+x)
    else:
        a = wk*h; A = np.array([[np.cos(a), np.sin(a)/wk], [-wk*np.sin(a), np.cos(a)]])
    e = np.exp(-gam*dt); O = np.diag([1, e]); Q0 = np.diag([0, 1-e*e])
    M = B@A@O@A@B; N = B@A
    return solve_discrete_lyapunov(M, N@Q0@N.T)
def run(nu, P, dt, cay, tau0=100.0, T=300.0):
    wP = P*T*0.69503476*c; w = nu*c; ks = range(P)
    wk = [2*wP*np.sin(np.pi*k/P) for k in ks]
    S = [sig(w, wk[k], 1/tau0 if k == 0 else 2*wk[k], dt, cay) for k in ks]
    Q = [s[0, 0] for s in S]; Qx = [1/(wk[k]**2 + w*w) for k in ks]
    pp = np.mean([s[1, 1] for s in S]) - 1
    def est(Q):
        prim = 0.5 - sum(wk[k]**2/2*Q[k] for k in ks)/P + sum(w*w/2*Q[k] for k in ks)/P
        vir = 0.5/P + sum(w*w/2*Q[k] for k in ks)/P + w*w/(2*P)*sum(Q[k] for k in ks if k)
        return prim, vir
    (p, v), (px, vx) = est(Q), est(Qx)
    return np.array([p - v, p - px, v - vx, pp])
modes = [(3700, 2), (1600, 1), (600, 3)]
for cay in (False, True):
    print("BCOCB (Cayley)" if cay else "BAOAB (exact)")
    for P in (1, 8, 16, 32, 64):
        for dt in (0.5, 0.25):
            kTP = P*300*KB
            r = sum(n*run(nu, P, dt, cay) for nu, n in modes)/9
            e = r[:3]*3*kTP
            print(f"  P={P:2d} dt={dt}: prim-vir {e[0]:+.2e}  prim err {e[1]:+.2e}  vir err {e[2]:+.2e} Eh/atom;  T bias {300*r[3]:+.2f} K")
