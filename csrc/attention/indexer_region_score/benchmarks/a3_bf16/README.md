# Standalone BF16 IndexerRegionScore benchmark

This harness reproduces [A3_PERFORMANCE.md](../../A3_PERFORMANCE.md)
without importing the repository Torch extension. It needs NumPy, Torch,
torch_npu, C++17, CANN headers and a built libcust_opapi.so exposing
IndexerRegionScore. It calls the two-stage ACLNN API on the current NPU stream.

Run from this directory after sourcing the SDK and configuring its standard
OPP/library paths for the built custom operator. Keep all generated inputs,
temporary files and SDK logs in the authorized workspace. Recorded runs use
/workspace/jy and device 2. Pass the real library directory explicitly.

~~~bash
python3 -B bench.py --generate
python3 -B build_bridge.py --opapi-dir /path/to/built/vendor/op_api/lib
python3 -B bench.py --impl fused --mode full --phase validate
python3 -B bench.py --impl fused --mode ids --phase validate
python3 -B bench.py --impl fused --mode full --phase time
python3 -B bench.py --impl fused --mode ids --phase time
~~~

Native timing refuses to run unless full-score and current-mode precision
reports pass. Defaults are 20 warmups, 100 samples, KV lengths 2048/4096/8192.
Save each implementation's results before rebuilding/replacing its library.

The generator uses seed 20946, BF16 Q/W/K, identity pages, top_k=512,
region_size=8 and head_scale=1. FP64 golden operates on stored decoded BF16
values. Inputs, exact mathematical IDs, golden scores and checksums are saved.
Validation checks defined scores, IDs, FP32-score/ID consistency and repeat
bits. First-observation raw scores and IDs are saved too.

## FP32 composition

Use a fresh process with the normal SDK OPP registry. The isolated native
validation registry cannot dispatch ordinary mm/sort operators; do not carry
its custom OPP override into these runs.

~~~bash
python3 -B bench.py --impl small_fp32_topk --mode full --phase validate
python3 -B bench.py --impl small_fp32_topk --mode ids --phase validate
python3 -B bench.py --impl small_fp32_topk --mode ids --phase time
~~~

This casts Q/K to FP32 with HF32 disabled before mm, then runs ReLU,
signed weighting, head reduction, visibility, TopK and ID ordering.
It does not guarantee the operator's tie rule for arbitrary inputs.
Other implementation choices are diagnostics: failed precision is never an
equal-accuracy performance baseline.

Timing JSON saves every wall/event sample, distributions and allocator memory.
Wall time includes executor creation and launch/sync. Use profiler Task
Duration for pure device time. Compatibility evidence does not establish
A3/A5 hardware or full-model equivalence.
