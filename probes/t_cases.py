"""Compile-only probes: README NKI example, dynamic=True, float64, graph breaks. One case per process: python t_cases.py <case>."""
import glob
import sys
import time

import torch
import torch.nn as nn
from torch_neuronx.compiler import CompileOptions, compile_only

case = sys.argv[1]


def neffs():
    return len(glob.glob("/tmp/neff_cache/**/*.neff", recursive=True))


def report(msg):
    print(f"RESULT[{case}] {msg}", flush=True)


try:
    with compile_only(CompileOptions(target="trn2")):
        if case == "nki":
            # Verbatim kernel from the README "Using NKI kernels" section, called under torch.compile.
            import nki
            import nki.isa as nisa
            import nki.language as nl
            from torch_neuronx import nki_op

            @nki.jit
            def add_kernel(a, b):
                a_sbuf = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=a_sbuf, src=a)
                b_sbuf = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=b_sbuf, src=b)
                c_sbuf = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=c_sbuf, data1=a_sbuf, data2=b_sbuf, op=nl.add)
                out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
                nisa.dma_copy(dst=out, src=c_sbuf)
                return out

            @nki_op("example::add", mutates_args={})
            def nki_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return add_kernel(a, b)

            f = torch.compile(lambda a, b: nki_add(a, b) * 2, backend="neuron", fullgraph=True)
            a = torch.rand(128, 128, device="neuron")
            out = f(a, a)
            report(f"no exception; out {tuple(out.shape)} {out.dtype}")

        elif case == "dynamic":
            m = nn.Linear(16, 8).to("neuron")
            f = torch.compile(m, backend="neuron", dynamic=True)
            for n in (4, 7):
                out = f(torch.randn(n, 16, device="neuron"))
            report(f"no exception with dynamic=True; last out {tuple(out.shape)}")

        elif case == "float64":
            m = nn.Linear(16, 8).double().to("neuron")
            x = torch.randn(4, 16, dtype=torch.float64, device="neuron")
            out = torch.compile(m, backend="neuron")(x)
            report(f"no exception; out dtype reported as {out.dtype}")

        elif case == "graphbreak":
            def fn(x):
                y = x * 2
                print("side effect")  # forces a Dynamo graph break
                return y + 1
            out = torch.compile(fn, backend="neuron")(torch.randn(8, 8, device="neuron"))
            report(f"no exception; out {tuple(out.shape)}")
except Exception as e:
    report(f"raised {type(e).__name__}: {str(e).splitlines()[0][:220]}")

t0 = time.time()
time.sleep(2)
report(f"neff files: {neffs()}")
