import sys, re, torch
sys.argv = [sys.argv[0]]  # run harness module definitions only
exec(open("/p/shlo_check.py").read().split("want = set(sys.argv[1:])")[0])
opts = CompileOptions(target="trn2")
for name, model, inputs, train in cases():
    if name not in ("mlp_train", "int64_math", "float64"): continue
    for label, gm, ex in capture(model, *inputs, train=train):
        with compile_only(opts):
            t = str(fx_to_stablehlo(gm, ex, opts)._mlir_module)
        sig = re.search(r"func\.func @main\((.*?)\)\s*->", t, re.S).group(1)
        print(f"== {name}/{label} SIG:", re.sub(r"loc\([^)]*\)", "", sig)[:500])
        for m in re.findall(r".*AwsNeuronErf.*", t)[:1]: print("ERF:", re.sub(r"loc\(.*", "", m)[:400])
