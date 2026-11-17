# A3 BF16 IndexerRegionScore performance

These are **910B2C compatibility measurements**, not A3 hardware results.
The standalone A3/compatibility device objects match bytewise.
The [machine-readable record](A3_PERFORMANCE.json) contains accuracy results,
timing samples, PMU summaries and object hashes.

## Workload and accuracy gate

BF16 Q=[1016,4,4,256], W=[1016,4,4], logical K=[639,1,256],
paged K=[639,1,1,256]. One request, identity pages, top_k=512,
region_size=8, head_scale=1; actual KV lengths 2048/4096/8192.
Visibility is clamped to 639 regions. Seed 20946; CANN 8.5.0,
Torch 2.9.0+cpu, torch_npu 2.9.0, device 2.

Independent FP64 scores use the stored, decoded BF16 values.
Before timing, both full and IDs-only modes run three times:
atol=1e-4/rtol=1e-5, exact mathematical TopK IDs, repeat-bit checks
and selection consistency against the actual FP32 scores.

All 5,197,856 defined FP32 score words match the previous A3 kernel
bitwise. Every TopK ID matches that kernel and FP64. Maximum score
absolute error is 3.1604795367456973e-6 at each length. Masked scores
are undefined and excluded.

Twenty-seven core/shape/long-history/boundary/view cases pass.
Across these plus four stress cases, 31 cases match the previous
A3 score bits and both ID modes. Stress cases retain the FP64
failures in [A3.md](A3.md), including cancellation error up to
0.4912499483 and near-tie/subnormal ID differences; these are not
precision passes.

## Adjacent timing runs

Median of 100 synchronized single calls after 20 warmups.
Wall time includes fresh ACLNN executor creation, launch and sync;
native scratch and output tensors are preallocated.

| KV length | Previous IDs-only ms | Optimized IDs-only ms | Speedup | Previous full ms | Optimized full ms |
| --------- | --------------------: | --------------------: | ------: | ---------------: | ----------------: |
| 2048 | 2.3959 | 0.2902 | 8.26x | 2.4213 | 0.2694 |
| 4096 | 5.1570 | 0.4695 | 10.98x | 5.1899 | 0.4157 |
| 8192 | 27.5725 | 0.6160 | 44.76x | 27.5755 | 0.5872 |

The eager composition casts BF16 Q/K to FP32, then runs mm, ReLU,
signed weighting, head reduction, causal masking, TopK and ID sort.
HF32 is disabled; scores and IDs pass this workload's accuracy gate.
It does not guarantee smaller-ID preference for arbitrary ties.

| KV length | FP32 composition IDs-only ms | Optimized speedup |
| --------- | ----------------------------: | ----------------: |
| 2048 | 31.4331 | 108.32x |
| 4096 | 46.3666 | 98.76x |
| 8192 | 50.0429 | 81.24x |

BF16-mm-output composition failed score accuracy and changed 8K IDs.
It is excluded from equal-accuracy performance comparisons.

## Bottleneck verification

At 8K, the previous AIC time was 7.021 ms, AIV 27.377 ms and device
Task Duration 27.507 ms. AIV scalar/vec ratios were 0.752/0.003.
Replacing final ID heap sorting with membership-order output reduced
wall time to about 15.94 ms while AIC stayed at 7.02 ms, isolating
a large scalar-selection cost.

MTE2 ratio 0.937 alone did not prove HBM saturation. A pure-memory
diagnostic retained Q/K addressing, clearing and ND-to-NZ copies,
removing MMAD, Fixpipe, vector consumption and selection. At 8K it
took 6.620 ms AIC time versus 7.021 ms in the operator: useful-payload
bandwidth 207.1 versus 195.3 GB/s. A contiguous-copy diagnostic took
0.5555 ms. Diagnostic outputs are invalid and cannot be used for inference.

Warm L2 reads were nearly 100% hits. SDK L2 size is 192 MiB;
Q/K/W use about 8.28 MiB and a conservative active set including
bridge/scores/IDs is about 29.2 MiB, below two-thirds of L2.
Raw ND-to-NZ PMU counters and useful bytes have different scopes;
neither is presented as physical HBM traffic. MAC and Fixpipe
ratios were low. Assuming 24 Cubes, 1.8 GHz and 16x16x16 FMA/cycle
gives estimated old padded AIC MFU about 0.86%; this peak denominator
is SDK-based, not calibrated against runtime clocks.

## Implementation and remaining work

1. Bundle up to four same-request H=4 queries into M=16; split
   request and token-batch tails.
2. Coalesce adjacent physical key rows; cache verified contiguous
   S=1 page prefixes per request. Invalid/noncontiguous pages fall back.
3. For visible<=1024, vector-sort only to obtain the Kth score,
   then compare and compact IDs in logical order. Internal sort
   tie order never determines the output.
4. Reuse dead scoring UB for selection and vectorize short-history IDs.
   Larger histories retain the bounded heap path.

A short-mask regression rejected the first compaction candidate before
timing. The final implementation uses 64-element alignment and passes it.

| KV length | Old device Task us | Optimized Task us | Optimized AIC us | Optimized AIV us |
| --------- | -----------------: | ----------------: | ---------------: | ---------------: |
| 2048 | 2253.488 | 175.104 | 143.283 | 173.102 |
| 4096 | 5029.568 | 310.888 | 274.905 | 308.816 |
| 8192 | 27507.056 | 528.680 | 347.300 | 525.617 |

Task Duration is the median of three profiler executions. NPU Event
intervals include enqueue gaps and occasional 0.16-us outliers, so they
are not used as pure-kernel latency. Remaining work is AIV execution
and mixed-core synchronization; no theoretical hardware limit is claimed.

## Memory and validation limits

Host/Tiling workspace formulas and FP32 quartet arithmetic are unchanged.
Extra Torch allocator peak above shared inputs remains 39,683,072 B
for full mode and 38,651,904 B for IDs-only: **0 B growth** at each
length. This includes explicit workspace/outputs, excluding private
driver/CANN pools. Explicit UB stays 54,016 B, reusing 24 KiB.

Standalone A3 and compatibility builds pass. Three focused
Ascend910_9391 CA cases cover request page-base changes, ties and
partial masks; defined scores and both ID modes match compatibility
hardware bitwise. The SDK maps this simulator to Ascend910B1.
Forced IDs-only batching still passes 14+3 tokens with 254804992 B
complete executor workspace.

A3 hardware, A5/A3 bit equivalence, full repository/Torch integration
and model precision remain unverified. The native measurements use
ACLNN through the standalone bridge, not the repository Torch binding.

## Reproduction

See [the standalone harness](benchmarks/a3_bf16/README.md), which
generates common inputs/FP64 golden, gates native timing on precision,
and records all samples and allocator memory.

The delivered source/build/tests/raw-output archive SHA256 is:
4702ce0ad51dfc17c1f4d97911f6a38f7f6beea6a2f069c27927cfd35b254863.
