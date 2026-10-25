"""Compare docs/torch_api_compatibility.md claims with observed CPU fallback, in compile-only mode.

For each documented function in a few sections, find an argument pattern that works on CPU, run the same
call on neuron, and read torch_neuronx.get_fallback_ops(). Only two outcomes count as mismatches:
  DOC_NATIVE_BUT_FELL_BACK   doc says Y and "Native", but a CPU fallback was recorded
  DOC_FALLBACK_BUT_NATIVE    doc says CPU_FALLBACK, but no fallback was recorded
Errors on neuron are reported as INCONCLUSIVE: compile-only feeds placeholder data, so data-dependent ops can fail.
"""
import re
import signal
import sys

import torch
import torch.nn.functional  # noqa: F401
import torch_neuronx
from torch_neuronx.compiler import CompileOptions, compile_only

DOC, SECTIONS = sys.argv[1], ("torch APIs", "torch.nn.functional", "torch.linalg", "torch.special")
SKIP = re.compile(r"(^torch\.(set_|use_|manual_seed|seed|initial_seed|compile|save|load|from_|sparse|quantiz|cuda|backends|"
                  r"distributed|autograd|jit|export|default|empty|frombuffer|asarray|as_tensor|is_|get_|nonzero_static))")

rows, section = [], None
for line in open(DOC):
    if line.startswith("## "):
        section = line[3:].strip()
    m = re.match(r"^\|\d+\|([A-Za-z0-9_.]+)\|(Y|N|CPU_FALLBACK)\|(.*)\|\s*$", line)
    if m and section in SECTIONS and not SKIP.search(m.group(1)) and not m.group(3).startswith("Host-only"):
        rows.append((m.group(1), m.group(2), m.group(3)))


def resolve(path):
    obj = torch
    for part in path.split(".")[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if callable(obj) else None


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def make(dev):
    torch.manual_seed(0)
    x = torch.randn(8, 8)
    sq = x @ x.T + 8 * torch.eye(8)
    return {k: v.to(dev) for k, v in {"x": x, "pos": x.abs() + 0.1, "sq": sq, "idx": torch.tensor([0, 3, 5])}.items()}


PATTERNS = [("x",), ("pos",), ("x", "x"), ("sq",), ("x", 1), ("x", 0), ("sq", "sq"), ("x", "idx"), ("x", "x", "x")]
cpu = make("cpu")
results = {}
with compile_only(CompileOptions(target="trn2")):
    neu = make("neuron")
    torch.neuron.synchronize()
    for name, doc, lim in rows:
        f = resolve(name)
        if f is None:
            results[name] = (doc, "UNRESOLVED", "")
            continue
        outcome = ("NO_PATTERN", "")
        for pat in PATTERNS:
            args_cpu = [cpu[a] if isinstance(a, str) else a for a in pat]
            try:
                signal.alarm(20); f(*args_cpu); signal.alarm(0)
            except Exception:
                signal.alarm(0)
                continue
            args_n = [neu[a] if isinstance(a, str) else a for a in pat]
            torch_neuronx.clear_op_tracking()
            try:
                signal.alarm(300); f(*args_n); torch.neuron.synchronize(); signal.alarm(0)
                fb = sorted(set(torch_neuronx.get_fallback_ops()))
                outcome = ("FELL_BACK" if fb else "NATIVE", ",".join(fb)[:150])
            except Exception as e:
                signal.alarm(0)
                fb = sorted(set(torch_neuronx.get_fallback_ops()))
                outcome = ("INCONCLUSIVE", f"{type(e).__name__}: {str(e).splitlines()[0][:100]} fb={','.join(fb)[:80]}")
            break
        results[name] = (doc, *outcome, lim[:90])
        print(f"ROW {name} | doc={doc} | observed={outcome[0]} | {outcome[1]}", flush=True)

mismatch = {"DOC_NATIVE_BUT_FELL_BACK": [], "DOC_FALLBACK_BUT_NATIVE": []}
for name, r in results.items():
    doc, obs = r[0], r[1]
    if doc == "Y" and r[-1].startswith("Native") and obs == "FELL_BACK":
        mismatch["DOC_NATIVE_BUT_FELL_BACK"].append((name, r[2]))
    if doc == "CPU_FALLBACK" and obs == "NATIVE":
        mismatch["DOC_FALLBACK_BUT_NATIVE"].append((name, r[-1]))
tested = sum(1 for r in results.values() if r[1] in ("NATIVE", "FELL_BACK"))
print(f"SUMMARY rows={len(rows)} tested_conclusively={tested} "
      f"inconclusive={sum(1 for r in results.values() if r[1] == 'INCONCLUSIVE')} "
      f"no_pattern={sum(1 for r in results.values() if r[1] in ('NO_PATTERN', 'UNRESOLVED'))}")
for k, v in mismatch.items():
    print(f"SUMMARY {k}: {len(v)}")
    for name, extra in v:
        print(f"MISMATCH {k} {name} :: {extra}")
