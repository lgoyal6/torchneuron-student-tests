"""Compile a GPT-2-architecture model (random weights, no download) for trn2 in compile-only mode.

Usage: python t_gpt2.py <n_layer> <seq_len>. Prints wall times and NEFF counts, waiting for async compiles.
"""
import glob
import sys
import time

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from torch_neuronx.compiler import CompileOptions, compile_only

n_layer, seq = int(sys.argv[1]), int(sys.argv[2])
cfg = GPT2Config(n_layer=n_layer)  # GPT-2 small dims: 768 hidden, 12 heads, 50257 vocab
model = GPT2LMHeadModel(cfg).eval()
params = sum(p.numel() for p in model.parameters()) / 1e6


def neffs():
    return glob.glob("/tmp/neff_cache/**/*.neff", recursive=True)


t0 = time.time()
with compile_only(CompileOptions(target="trn2")), torch.no_grad():
    m = model.to("neuron")
    f = torch.compile(m, backend="neuron", fullgraph=True, dynamic=False)
    ids = torch.randint(0, cfg.vocab_size, (1, seq), device="neuron")
    out = f(input_ids=ids, use_cache=False).logits
returned = time.time() - t0
print(f"RESULT gpt2 {n_layer}L {params:.0f}M params seq={seq}: returned in {returned:.0f}s, logits {tuple(out.shape)}, neffs at return {len(neffs())}", flush=True)

# Compiles finish asynchronously; wait until the NEFF count stops changing.
last, stable = -1, 0
while stable < 3:
    time.sleep(10)
    n = len(neffs())
    stable = stable + 1 if n == last else 0
    last = n
sizes = sorted((__import__("os").path.getsize(p) for p in neffs()), reverse=True)
print(f"RESULT gpt2 {n_layer}L: all compiles done after {time.time()-t0:.0f}s, {last} neffs, largest {sizes[0]/1e6:.1f} MB", flush=True)
