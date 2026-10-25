import sys, re, traceback, torch
sys.argv = [sys.argv[0]]
exec(open("/p/shlo_check.py").read().split("want = set(sys.argv[1:])")[0])
opts = CompileOptions(target="trn2")
for name, model, inputs, train in cases():
    if name not in ("cumsum_sort_topk", "argmax_embedding", "llama_2l_eager_attn", "gpt2_2l_sdpa"): continue
    for label, gm, ex in capture(model, *inputs, train=train):
        try:
            with compile_only(opts):
                t = str(fx_to_stablehlo(gm, ex, opts)._mlir_module)
        except Exception as e:
            print(f"== {name} LOWERING ERROR {type(e).__name__}:", str(e)[:1500].replace("\n", " | "))
            continue
        for m in sorted(set(re.sub(r"loc\(.*", "", l).strip() for l in re.findall(r".*custom_call @AwsNeuron(?:Silu|TopK|ArgMax).*", t)))[:3]:
            print(f"== {name}: {m[:500]}")
