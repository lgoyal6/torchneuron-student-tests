"""Mirror of the repo's test_cpu_inputs_fail_loud: CPU input to a neuron-compiled model under compile-only."""
import torch
import torch.nn as nn
from torch_neuronx.compiler import CompileOptions, compile_only

with compile_only(CompileOptions(target="trn2")):
    m = nn.Sequential(nn.Linear(8, 8), nn.ReLU()).to("neuron")
    x_cpu = torch.randn(4, 8)  # deliberately not moved to neuron
    try:
        with torch.no_grad():
            torch.compile(m, backend="neuron")(x_cpu)
        print("STAGE compiled call: ok")
    except Exception as e:
        print(f"STAGE compiled call: RAISED {type(e).__name__}: {str(e)[:500]}")
    try:
        torch.neuron.synchronize()
        print("STAGE synchronize: ok")
    except Exception as e:
        print(f"STAGE synchronize: RAISED {type(e).__name__}: {str(e)[:500]}")
