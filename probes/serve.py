"""Tiny next-token service on the neuron backend, following the README's advice to bucket shapes.

At startup it compiles one graph per sequence bucket (compile-only here, since there is no device),
then /next pads the prompt to a bucket and returns the argmax next token.
"""
import copy
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Config, GPT2LMHeadModel
from torch_neuronx.compiler import CompileOptions, compile_only

BUCKETS = (32, 64)
cfg = GPT2Config(n_layer=2)
torch.manual_seed(0)
cpu_model = GPT2LMHeadModel(cfg).eval()
app = FastAPI()
state = {}


@app.on_event("startup")
def warm():
    t0 = time.time()
    state["ctx"] = compile_only(CompileOptions(target="trn2"))
    state["ctx"].__enter__()
    neuron_model = copy.deepcopy(cpu_model).to("neuron")  # same weights as the CPU reference
    state["model"] = torch.compile(neuron_model, backend="neuron", dynamic=False)
    with torch.no_grad():
        for b in BUCKETS:
            state["model"](input_ids=torch.zeros(1, b, dtype=torch.long, device="neuron"), use_cache=False)
    state["warm_s"] = time.time() - t0


class Req(BaseModel):
    ids: list[int]


@app.post("/next")
def next_token(r: Req):
    b = next(x for x in BUCKETS if x >= len(r.ids))
    ids = torch.tensor([r.ids + [0] * (b - len(r.ids))], device="neuron")
    with torch.no_grad():
        logits = state["model"](input_ids=ids, use_cache=False).logits[0, len(r.ids) - 1]
        ref = cpu_model(input_ids=torch.tensor([r.ids]), use_cache=False).logits[0, -1]
    return {
        "bucket": b,
        "neuron_next_token": int(logits.cpu().argmax()),
        "neuron_logits_sample": [round(float(v), 4) for v in logits.cpu()[:4]],
        "cpu_reference_next_token": int(ref.argmax()),
        "warmup_seconds": round(state["warm_s"], 1),
    }
