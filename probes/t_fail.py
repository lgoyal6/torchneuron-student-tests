"""Where does a compile failure surface in compile-only mode? python t_fail.py <case>

Cases: ok (control), badflag (NEURON_CC_FLAGS set by caller), eigh, fft, sort (ops the compiler may reject).
For each, records whether the compiled call, torch.neuron.synchronize(), or get_neff_info() raises.
"""
import sys

import torch
import torch.nn as nn
import torch_neuronx
from torch_neuronx.compiler import CompileOptions, compile_only

case = sys.argv[1]
fns = {
    "ok": lambda x: torch.relu(x @ x.T),
    "badflag": lambda x: torch.relu(x @ x.T),
    "eigh": lambda x: torch.linalg.eigh(x @ x.T)[0],
    "fft": lambda x: torch.fft.fft(x).real,
    "sort": lambda x: torch.sort(x, dim=-1)[0],
}


def stage(name, f):
    try:
        r = f()
        print(f"STAGE[{case}] {name}: ok", flush=True)
        return r
    except Exception as e:
        print(f"STAGE[{case}] {name}: RAISED {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
        return None


with compile_only(CompileOptions(target="trn2")):
    x = torch.randn(64, 64, device="neuron")
    out = stage("compiled call", lambda: torch.compile(fns[case], backend="neuron", fullgraph=True)(x))
    stage("synchronize", torch.neuron.synchronize)
    art = getattr(out, "_neff_artifact", None) if out is not None else None
    print(f"STAGE[{case}] artifact attached: {art is not None}", flush=True)
    if art is not None:
        info = stage("get_neff_info", lambda: torch_neuronx._C.get_neff_info(art.cache_key))
        print(f"STAGE[{case}] neff_info: {str(info)[:160]}", flush=True)
print(f"STAGE[{case}] script reached end", flush=True)
