"""Check torch-neuronx's FX -> StableHLO lowering numerically, on CPU, with no Neuron hardware.

For each case: capture the post-AOT forward graph the neuron backend lowers (same helper as the repo's
unit tests), lower it with the public torch_neuronx.compiler.fx_to_stablehlo, execute the StableHLO on
XLA's CPU backend (via JAX), and compare against running the same graph, and the original module, in PyTorch.
Usage: python shlo_check.py <case> [<case> ...]
"""
import sys

import numpy as np
import torch
import torch.nn as nn
from torch._dynamo.backends.common import aot_autograd
from torch_neuronx.compiler import CompileOptions, compile_only, fx_to_stablehlo
from torch_neuronx.neuron_dynamo_backend.decompositions import get_compile_decomposition_table


def capture(model, *inputs, train=False):
    """Return [(label, gm, real_inputs)] for the forward graph, plus the backward graph when train=True."""
    box = {}

    def make_compiler(label):
        def compiler(gm, example_inputs):
            def run(*args):
                # Compile-time example_inputs are fake tensors; record the real ones at call time.
                flat = args[0] if len(args) == 1 and isinstance(args[0], list) else args
                box[label] = (gm, [a.detach().clone() if isinstance(a, torch.Tensor) else a for a in flat])
                return gm(*flat)
            return run
        return compiler

    aot = aot_autograd(fw_compiler=make_compiler("fwd"), bw_compiler=make_compiler("bwd"),
                       decompositions=get_compile_decomposition_table())
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=aot, fullgraph=True)
    if train:
        out = compiled(*inputs)
        out = out.logits if hasattr(out, "logits") else out
        out.float().pow(2).mean().backward()
    else:
        with torch.no_grad():
            compiled(*inputs)
    return [(k, *box[k]) for k in ("fwd", "bwd") if k in box]


def xla_cpu_run(mlir_text, args):
    import jax
    jax.config.update("jax_enable_x64", True)
    from jax._src import xla_bridge as xb
    from jax._src.lib import xla_client as xc

    backend = xb.get_backend("cpu")
    opts = xc.CompileOptions()
    devs = backend.devices()[:1]
    try:
        exe = backend.compile_and_load(mlir_text, xc.DeviceList(tuple(devs)), opts)
    except AttributeError:
        exe = backend.compile(mlir_text, xc.DeviceList(tuple(devs)), opts)
    bufs = [jax.device_put(np.ascontiguousarray(a), devs[0]) for a in args]
    res = exe.execute_sharded(bufs)  # jaxlib 0.12: a flat sequence of single-device arrays
    outs = [o[0] for o in res.disassemble_into_single_device_arrays()]
    return [np.asarray(o) for o in outs]


def to_np(t):
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        import ml_dtypes
        return t.float().numpy().astype(ml_dtypes.bfloat16)
    return t.numpy()


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(64, 128), nn.Linear(128, 10)

    def forward(self, x):
        return self.b(torch.nn.functional.gelu(self.a(x)))


def cases():
    """(name, model, inputs, train)."""
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    F = torch.nn.functional
    yield "negative_control", MLP().eval(), (torch.randn(32, 64),), False
    yield "mlp_gelu", MLP().eval(), (torch.randn(32, 64),), False
    yield "layernorm", nn.LayerNorm(256).eval(), (torch.randn(8, 256) * 3 + 1,), False
    yield "softmax", nn.Softmax(dim=-1), (torch.randn(8, 1000) * 5,), False
    yield "conv2d_bn_pool", nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2)).eval(), (torch.randn(2, 3, 32, 32),), False
    yield "cumsum_sort_topk", Lambda(lambda x: torch.cumsum(x, 1) + torch.sort(x, 1)[0] + torch.topk(x, 64, 1)[0].sum(1, keepdim=True)), (torch.randn(4, 128),), False
    yield "gather_index", Lambda(lambda x, i: x[:, i[0]] * 2 + torch.gather(x, 1, i.expand(x.shape[0], -1))), (torch.randn(8, 100), torch.randint(0, 100, (1, 20))), False
    yield "scatter_add_where", Lambda(lambda x, i: torch.where(x > 0, x, 0.1 * x).scatter_add(1, i.expand(x.shape[0], -1), x[:, :20])), (torch.randn(8, 100), torch.randint(0, 100, (1, 20))), False
    yield "cross_entropy", Lambda(lambda x, y: F.cross_entropy(x, y)), (torch.randn(16, 1000), torch.randint(0, 1000, (16,))), False
    yield "int64_math", Lambda(lambda i: (i * 3 + 7) // 2 % 1000 + i.cumsum(1)), (torch.randint(0, 10**6, (4, 64)),), False
    yield "float64", Lambda(lambda x: torch.exp(x) * 1e-3 + x.double().sin()), (torch.randn(8, 8, dtype=torch.float64),), False
    yield "bf16_matmul", Lambda(lambda a, b: (a @ b).float()), (torch.randn(64, 256).bfloat16(), torch.randn(256, 64).bfloat16()), False
    yield "argmax_embedding", nn.Sequential(nn.Embedding(1000, 64), Lambda(lambda e: e.sum(-1).argmax(-1))), (torch.randint(0, 1000, (4, 32)),), False
    yield "mlp_train", MLP().train(), (torch.randn(32, 64),), True
    yield "gpt2_2l_eager_attn", GPT2LMHeadModel(GPT2Config(n_layer=2, attn_implementation="eager")).eval(), (torch.randint(0, 50257, (1, 64)),), False
    yield "gpt2_2l_sdpa", GPT2LMHeadModel(GPT2Config(n_layer=2, attn_implementation="sdpa")).eval(), (torch.randint(0, 50257, (1, 64)),), False
    yield "gpt2_2l_train", GPT2LMHeadModel(GPT2Config(n_layer=2, attn_implementation="eager", resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).train(), (torch.randint(0, 50257, (1, 64)),), True
    yield "gpt2_12l_eager_attn", GPT2LMHeadModel(GPT2Config(n_layer=12, attn_implementation="eager")).eval(), (torch.randint(0, 50257, (1, 128)),), False
    yield "llama_2l_eager_attn", LlamaForCausalLM(LlamaConfig(hidden_size=512, intermediate_size=1376, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=4, vocab_size=32000, attn_implementation="eager")).eval(), (torch.randint(0, 32000, (1, 64)),), False
    yield "llama_2l_train", LlamaForCausalLM(LlamaConfig(hidden_size=512, intermediate_size=1376, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=4, vocab_size=32000, attn_implementation="eager")).train(), (torch.randint(0, 32000, (1, 64)),), True


class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, *a):
        return self.f(*a)


def flat_outputs(o):
    o = o if isinstance(o, (list, tuple)) else [o]
    return [to_np(t) for t in o if isinstance(t, torch.Tensor)]


NP = {"f32": np.float32, "f64": np.float64, "bf16": __import__("ml_dtypes").bfloat16, "f16": np.float16, "i64": np.int64, "i32": np.int32,
      "i16": np.int16, "i8": np.int8, "ui8": np.uint8, "i1": np.bool_}


def prepare(text):
    """Make the module runnable on XLA CPU. Returns (text, notes)."""
    import re
    notes = []
    # Input/output buffer aliasing hints: runtime-memory optimizations, no effect on values.
    t2 = re.sub(r"tf\.aliasing_output = \d+ : i32,?\s*", "", text)
    t2 = re.sub(r"mhlo\.input_output_alias = \[[^\]]*\],?\s*", "", t2)
    t2 = re.sub(r"jax\.buffer_donor = true,?\s*", "", t2)
    if t2 != text:
        notes.append("stripped aliasing hints")
    # AwsNeuronErf is elementwise erf; swap in the standard CHLO op so XLA CPU can run it.
    t3, n = re.subn(r"stablehlo\.custom_call @AwsNeuronErf\((%[\w#]+)\)\s*(\{[^}]*\})?\s*:\s*\((tensor<[^>]+>)\)\s*->\s*(tensor<[^>]+>)",
                    r"chlo.erf \1 : \3 -> \4", t2)
    if n:
        notes.append(f"replaced {n} AwsNeuronErf with chlo.erf")
    t3 = substitute_kernels(t3, notes)
    return t3, notes


def substitute_kernels(text, notes):
    """Swap AwsNeuronSilu/TopK/ArgMax custom calls for standard StableHLO/CHLO with the same meaning."""
    import re
    out, counts = [], {}
    pat = re.compile(r'^(\s*)(%[\w]+)(?::2)? = stablehlo\.custom_call @AwsNeuron(Silu|TopK|ArgMax)\((%[\w#]+)\)\s*'
                     r'(?:\{backend_config = "(\d+)"\})?\s*:\s*\((tensor<([^>]+)>)\)\s*->\s*(.*?)(\s+loc\(.*)?$')
    renames = {}
    for line in text.split("\n"):
        m = pat.match(line)
        if not m:
            out.append(line)
            continue
        ind, res, kind, x, cfg, xt, xshape, rt = m.group(1, 2, 3, 4, 5, 6, 7, 8)
        dims = xshape.split("x")[:-1]
        elt = xshape.split("x")[-1]
        r = "sub" + res[1:]
        if kind == "Silu":
            out.append(f"{ind}%{r}_sg = stablehlo.logistic {x} : {xt}")
            out.append(f"{ind}{res} = stablehlo.multiply {x}, %{r}_sg : {xt}")
        elif kind == "TopK":
            k = int(cfg)
            vt = f"tensor<{'x'.join(dims[:-1] + [str(k)])}x{elt}>"
            it32 = f"tensor<{'x'.join(dims[:-1] + [str(k)])}xi32>"
            it64 = it32.replace("xi32>", "xi64>")
            out.append(f"{ind}%{r}_tk:2 = chlo.top_k({x}, k = {k}) : {xt} -> ({vt}, {it32})")
            out.append(f"{ind}%{r}_i = stablehlo.convert %{r}_tk#1 : ({it32}) -> {it64}")
            renames[f"{res}#0"], renames[f"{res}#1"] = f"%{r}_tk#0", f"%{r}_i"
        elif kind == "ArgMax":
            axis = int(cfg)
            if axis != len(dims) - 1:
                out.append(line)
                continue
            vt = f"tensor<{'x'.join(dims[:-1] + ['1'])}x{elt}>"
            it32 = f"tensor<{'x'.join(dims[:-1] + ['1'])}xi32>"
            it64 = it32.replace("xi32>", "xi64>")
            out.append(f"{ind}%{r}_tk:2 = chlo.top_k({x}, k = 1) : {xt} -> ({vt}, {it32})")
            out.append(f"{ind}%{r}_c = stablehlo.convert %{r}_tk#1 : ({it32}) -> {it64}")
            out.append(f"{ind}{res} = stablehlo.reshape %{r}_c : ({it64}) -> {rt.strip()}")
        counts[kind] = counts.get(kind, 0) + 1
    text = "\n".join(out)
    for old, new in renames.items():
        text = re.sub(re.escape(old) + r"\b", new, text)
    if counts:
        notes.append("substituted reference impls for " + ", ".join(f"{v}x AwsNeuron{k}" for k, v in counts.items()))
    return text


def arg_types(text):
    import re
    sig = re.search(r"func\.func (?:public )?@main\((.*?)\)\s*->", text, re.S).group(1)
    return re.findall(r"%arg\d+: tensor<(?:[^>]*x)?([a-z]+\d+)>", sig)


def check(tag, name, gm, ex):
    with compile_only(opts):
        art = fx_to_stablehlo(gm, ex, opts)
    text = str(art._mlir_module)
    n_custom = text.count("stablehlo.custom_call")
    tens = [t for t in ex if isinstance(t, torch.Tensor)]
    spec_in = [tuple(s.shape) for s in art.io_spec.inputs]
    ex_in = [tuple(t.shape) for t in tens]
    if spec_in != ex_in:
        print(f"{tag} UNMAPPABLE inputs (io_spec n={len(spec_in)} vs graph n={len(ex_in)}): cannot compare", flush=True)
        return
    text, notes = prepare(text)
    n_custom = text.count("stablehlo.custom_call")
    if n_custom:
        names = sorted(set(__import__("re").findall(r"custom_call @([A-Za-z0-9_.]+)", text)))
        print(f"{tag} SKIP: {n_custom} custom_call(s) {names} are Neuron kernels, not runnable on CPU", flush=True)
        return
    with torch.no_grad():
        ref = flat_outputs(gm(*ex))
    if name == "negative_control":
        text = text.replace("stablehlo.add", "stablehlo.subtract", 1)
    args = [to_np(t) for t in tens]
    for i, (a, ty) in enumerate(zip(args, arg_types(text))):
        want_dt = NP.get(ty)
        if want_dt is not None and a.dtype != want_dt:
            notes.append(f"input {i} {a.dtype}->{np.dtype(want_dt)}")
            args[i] = a.astype(want_dt)
    got = xla_cpu_run(text, args)
    if len(got) != len(ref):
        print(f"{tag} OUTPUT COUNT {len(got)} vs {len(ref)}", flush=True)
        return
    worst, dtypes = 0.0, set()
    for g, r in zip(got, ref):
        dtypes.add(f"{r.dtype}->{g.dtype}" if g.dtype != r.dtype else str(r.dtype))
        if g.shape != r.shape:
            print(f"{tag} SHAPE MISMATCH {g.shape} vs {r.shape}", flush=True)
            return
        if r.dtype == np.bool_ or np.issubdtype(r.dtype, np.integer):
            err = float((g.astype(np.int64) != r.astype(np.int64)).mean())
        else:
            denom = max(float(np.abs(r.astype(np.float64)).max()), 1e-6)
            err = float(np.abs(g.astype(np.float64) - r.astype(np.float64)).max() / denom)
        worst = max(worst, err)
    verdict = "MATCH" if worst < 1e-4 else ("CLOSE" if worst < 1e-2 else "MISMATCH")
    print(f"{tag} {verdict} outputs={len(got)} max_rel_err={worst:.2e} dtypes={sorted(dtypes)} graph_inputs={len(tens)} notes={notes}", flush=True)


want = set(sys.argv[1:])
opts = CompileOptions(target="trn2")
for name, model, inputs, train in cases():
    if want and name not in want:
        continue
    try:
        graphs = capture(model, *inputs, train=train)
    except Exception as e:
        print(f"SHLO[{name}] CAPTURE ERROR {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
        continue
    for label, gm, ex in graphs:
        tag = f"SHLO[{name}/{label}]"
        try:
            check(tag, name, gm, ex)
        except Exception as e:
            print(f"{tag} ERROR {type(e).__name__}: {str(e).splitlines()[0][:220]}", flush=True)
