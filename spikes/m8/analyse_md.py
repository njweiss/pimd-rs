"""M8 spike Q3 analysis of a P = 1 pimd-rs run of NMA in water (numpy only).

    python3 spikes/m8/analyse_md.py RUN_DIR --box L [--skip-ps 2] [--dt 0.5]

Reads RUN_DIR/trajectory.xyz (P = 1, unwrapped, Å; atoms 0-11 NMA in build_box.py order, then
O H H per water), observables.csv and, if present, segments.csv. Prints one JSON object:
- NMA: per-bond max |r/r_0 - 1| over all frames (r_0 = frame 0), mean bond lengths, and the
  amide torsion omega (C0-C4-N6-C8) min |omega| over frames (trans = 180°).
- water: max O-H over all frames and molecules; frames in which some H's nearest O (minimum
  image) is not its own O (proton transfer / dissociation); max displacement of any atom from
  frame 0 (unwrapped).
- g_OO (water O only, minimum image, 0.02-0.5 L, bins 0.05 Å) averaged after --skip-ps: first peak
  position and height, first minimum.
- T_est mean and sd after --skip-ps (observables.csv), and per segment (H_end - H_start)/t_seg in
  Eh per atom per ps (segments.csv).
"""

import argparse
import csv
import json
import os

import numpy as np

NMA_BONDS = [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (6, 7), (6, 8), (8, 9), (8, 10), (8, 11)]
NMA_NAMES = ["C1-H", "C1-H", "C1-H", "C1-C2", "C2=O", "C2-N", "N-H", "N-C3", "C3-H", "C3-H", "C3-H"]


def frames(path):
    with open(path) as f:
        while True:
            head = f.readline()
            if not head:
                return
            n = int(head)
            meta = f.readline()
            step = int(meta.split("step=")[1].split()[0])
            xyz = np.array([[float(x) for x in f.readline().split()[1:4]] for _ in range(n)])
            yield step, xyz


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--box", type=float, required=True)
    ap.add_argument("--skip-ps", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=0.5)
    a = ap.parse_args()
    L = a.box
    traj = list(frames(os.path.join(a.run, "trajectory.xyz")))
    q0 = traj[0][1]
    n = len(q0)
    o_idx = np.arange(12, n, 3)
    h_idx = np.concatenate([o_idx + 1, o_idx + 2])
    own_o = np.concatenate([o_idx, o_idx])
    r0 = np.array([np.linalg.norm(q0[j] - q0[i]) for i, j in NMA_BONDS])
    max_rel = np.zeros(len(NMA_BONDS))
    bond_sum = np.zeros(len(NMA_BONDS))
    omega_min = 180.0
    oh_max = 0.0
    bad_frames = 0
    disp_max = 0.0
    edges = np.arange(0.0, 0.5 * L + 1e-9, 0.05)
    hist = np.zeros(len(edges) - 1)
    n_g = 0
    for step, q in traj:
        r = np.array([np.linalg.norm(q[j] - q[i]) for i, j in NMA_BONDS])
        max_rel = np.maximum(max_rel, np.abs(r / r0 - 1))
        bond_sum += r
        omega_min = min(omega_min, abs(dihedral(q[0], q[4], q[6], q[8])))
        d_oh = np.concatenate([np.linalg.norm(q[o_idx + 1] - q[o_idx], axis=1),
                               np.linalg.norm(q[o_idx + 2] - q[o_idx], axis=1)])
        oh_max = max(oh_max, d_oh.max())
        dho = q[h_idx][:, None, :] - q[o_idx][None, :, :]
        dho -= L * np.round(dho / L)
        nearest = o_idx[np.argmin((dho ** 2).sum(-1), axis=1)]
        bad_frames += int(np.any(nearest != own_o))
        disp_max = max(disp_max, np.linalg.norm(q - q0, axis=1).max())
        if step * a.dt / 1000.0 >= a.skip_ps:
            d = q[o_idx][:, None, :] - q[o_idx][None, :, :]
            d -= L * np.round(d / L)
            rr = np.sqrt((d ** 2).sum(-1))[np.triu_indices(len(o_idx), 1)]
            hist += np.histogram(rr, edges)[0]
            n_g += 1
    no = len(o_idx)
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    # Normalised to the water-O number density of the box (NMA excluded, so g -> slightly < 1).
    g = 2.0 * hist * L ** 3 / (n_g * no * (no - 1) * shell) if n_g else hist
    rc = 0.5 * (edges[1:] + edges[:-1])
    k1 = int(np.argmax(np.where(rc < 3.3, g, 0)))
    kmin = k1 + int(np.argmin(np.where((rc > rc[k1]) & (rc < 4.0), g, np.inf)[k1:]))
    out = {
        "run": a.run, "frames": len(traj), "last_step": traj[-1][0],
        "nma_max_rel_bond_change": dict(zip([f"{nm}({i}-{j})" for nm, (i, j) in zip(NMA_NAMES, NMA_BONDS)],
                                            np.round(max_rel, 4).tolist())),
        "nma_mean_bond_a": dict(zip([f"{nm}({i}-{j})" for nm, (i, j) in zip(NMA_NAMES, NMA_BONDS)],
                                    np.round(bond_sum / len(traj), 4).tolist())),
        "omega_min_abs_deg": round(omega_min, 2),
        "water_oh_max_a": round(float(oh_max), 4),
        "frames_with_h_not_nearest_own_o": bad_frames,
        "max_displacement_a": round(float(disp_max), 3),
        "goo_frames": n_g,
        "goo_peak_r_a": round(float(rc[k1]), 3), "goo_peak_g": round(float(g[k1]), 3),
        "goo_min_r_a": round(float(rc[kmin]), 3), "goo_min_g": round(float(g[kmin]), 3),
    }
    with open(os.path.join(a.run, "observables.csv")) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t_est_k"]) for r in rows if int(r["step"]) * a.dt / 1000 >= a.skip_ps])
    if len(t):
        out["t_est_mean_k"] = round(float(t.mean()), 2)
        out["t_est_sd_k"] = round(float(t.std()), 2)
    seg = os.path.join(a.run, "segments.csv")
    if os.path.exists(seg):
        with open(seg) as f:
            srows = list(csv.DictReader(f))
        out["segments"] = [{"segment": int(r["segment"]), "h_start": float(r["h_start_eh"]),
                            "dh_eh_per_atom": float(r["h_end_eh"]) - float(r["h_start_eh"])}
                           for r in srows]
    np.savetxt(os.path.join(a.run, "goo.csv"), np.column_stack([rc, g]), delimiter=",",
               header="r_angstrom,g_oo", comments="")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
