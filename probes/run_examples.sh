#!/bin/bash
# Run repo examples unmodified with TORCH_NEURONX_COMPILE_ONLY=1. Each entry: "<dir>|<script>|<args>".
export TORCH_NEURONX_COMPILE_ONLY=1
for ex in "gpt2-train-loop|train.py|--iterations 1" "simple_moe|simple_moe.py|" "llama|llama.py|" \
          "fused_attention|scaled_dot_product_attention.py|" "fused_attention|multihead_attention_forward.py|" \
          "compile|device_placement.py|" "compile|dtype_handling.py|" "compile|nki_integration.py|"; do
  IFS='|' read -r d script args <<< "$ex"
  rm -rf /tmp/neff_cache; t0=$(date +%s)
  (cd /repo/examples/$d && timeout 1200 python $script $args > /tmp/ex.log 2>&1); rc=$?
  echo "== $d/$script $args | exit=$rc | $(( $(date +%s)-t0 ))s | neffs=$(find /tmp/neff_cache -name '*.neff' 2>/dev/null | wc -l) | compile-fail lines=$(grep -ciE 'Compilation failed|Command failed' /tmp/ex.log)"
  grep -E "Error|error:|Traceback" /tmp/ex.log | grep -vE "NRT:|NEURON_RT_DBG" | tail -2 | cut -c1-220
done
