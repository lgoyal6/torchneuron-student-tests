import torch, torch_neuronx
from torch_neuronx.compiler import CompileOptions, compile_only
with compile_only(CompileOptions(target="trn2")):
    x = torch.randn(8, 8, device="neuron"); idx = torch.tensor([0, 3, 5], device="neuron")
    torch.neuron.synchronize()
    for name, f in [("sin", lambda: torch.sin(x)), ("heaviside", lambda: torch.heaviside(x, x)), ("take", lambda: torch.take(x, idx))]:
        torch_neuronx.clear_op_tracking()
        try:
            f(); torch.neuron.synchronize(); err = ""
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
        print(f"PROTO {name}: fallback={torch_neuronx.get_fallback_ops()} executed={torch_neuronx.get_executed_ops()[:5]} err={err}")
