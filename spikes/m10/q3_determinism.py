"""M10 spike Q3 (vi): are cueq evaluations bitwise repeatable (SPEC D-3 on CUDA)?

    pixi run -e ml spikes/m10/cueq/.venv/bin/python spikes/m10/q3_determinism.py VARIANT P [--det]

VARIANT e3nn | cueq, float32, the Q3 inputs. Ten evaluations of the same inputs in one process;
prints how many differ bitwise from the first (energies and forces) and the largest force
difference. --det sets torch.use_deterministic_algorithms(True) (the engine's default switch).
"""

import argparse
import json

import torch

import common

ap = argparse.ArgumentParser()
ap.add_argument("variant", choices=["e3nn", "cueq"])
ap.add_argument("p", type=int)
ap.add_argument("--det", action="store_true")
a = ap.parse_args()
dt = torch.float32
model = common.load_small()
z = model.atomic_numbers.tolist()
r_max = float(model.r_max)
if a.variant == "cueq":
    from mace.cli.convert_e3nn_cueq import run as convert
    dflt = torch.get_default_dtype()
    model = convert(model.to(dt), device="cuda", return_model=True)
    torch.set_default_dtype(dflt)
mod = common.wrap_eager(model, dt)
if a.det:
    torch.use_deterministic_algorithms(True)
args = common.inputs(a.p, dt, z, r_max)
res = {"variant": a.variant, "p": a.p, "deterministic_switch": a.det}
try:
    outs = []
    for _ in range(10):
        e, f, _ = mod(*args)
        outs.append((e.detach().clone(), f.detach().clone()))
    e0, f0 = outs[0]
    res["n_e_differ"] = sum(int(not torch.equal(e, e0)) for e, _ in outs[1:])
    res["n_f_differ"] = sum(int(not torch.equal(f, f0)) for _, f in outs[1:])
    res["max_df"] = max(float((f - f0).abs().max()) for _, f in outs[1:])
except Exception as ex:  # a result
    res["error"] = f"{type(ex).__name__}: {str(ex)[:600]}"
print(json.dumps(res))
