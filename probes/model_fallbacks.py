"""Which ops fall back to CPU when common HF architectures run on neuron (compile-only, random weights)?

Runs each model forward in eager mode and under torch.compile(backend="neuron"), then reports
torch_neuronx.get_fallback_ops(). Sizes are reduced (2 layers) since fallback is per-op, not per-size.
"""
import torch
import torch_neuronx
from torch_neuronx.compiler import CompileOptions, compile_only
from transformers import (BertConfig, BertModel, GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM,
                          ViTConfig, ViTModel)

models = {
    "gpt2": (GPT2LMHeadModel, GPT2Config(n_layer=2), "ids"),
    "llama": (LlamaForCausalLM, LlamaConfig(hidden_size=512, intermediate_size=1376, num_hidden_layers=2,
                                            num_attention_heads=8, num_key_value_heads=4, vocab_size=32000), "ids"),
    "bert": (BertModel, BertConfig(num_hidden_layers=2), "ids"),
    "vit": (ViTModel, ViTConfig(num_hidden_layers=2), "pix"),
}

with compile_only(CompileOptions(target="trn2")), torch.no_grad():
    for name, (cls, cfg, kind) in models.items():
        torch.manual_seed(0)
        m = cls(cfg).eval().to("neuron")
        inp = (torch.randint(0, 1000, (1, 64), device="neuron") if kind == "ids"
               else torch.randn(1, 3, 224, 224, device="neuron"))
        call = (lambda mod: mod(input_ids=inp)) if kind == "ids" else (lambda mod: mod(pixel_values=inp))
        for mode in ("eager", "compiled"):
            torch_neuronx.clear_op_tracking()
            try:
                call(m if mode == "eager" else torch.compile(m, backend="neuron", dynamic=False))
                torch.neuron.synchronize()
                err = ""
            except Exception as e:
                err = f" ERROR {type(e).__name__}: {str(e).splitlines()[0][:150]}"
            fb = sorted(set(torch_neuronx.get_fallback_ops()))
            print(f"MODEL {name} {mode}: {len(fb)} fallback ops {fb}{err}", flush=True)
