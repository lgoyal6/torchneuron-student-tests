"""Issue #1021: Tensor.to("neuron:1") on a neuron:0 tensor stays on neuron:0 with no error."""
import torch
from torch_neuronx.compiler import CompileOptions, compile_only
with compile_only(CompileOptions(target="trn2")):
    print("ISSUE device_count:", torch.neuron.device_count())
    a = torch.zeros(4, device="neuron:0")
    try:
        b = a.to("neuron:1")
        print("ISSUE a.to('neuron:1') ->", b.device)
    except Exception as e:
        print("ISSUE raised", type(e).__name__, str(e)[:150])
