#!/usr/bin/env python3
"""Native fused versus eager matmul composition, with independent FP64 validation."""

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu

ROOT = Path(__file__).resolve().parent
T, G, H, D, R, TOPK = 1016, 4, 4, 256, 639, 512
LENGTHS = (2048, 4096, 8192)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def generate():
    rng = np.random.default_rng(20946)
    q = torch.from_numpy(rng.uniform(-0.75, 0.75, (T, G, H, D)).astype(np.float32)).bfloat16()
    w = torch.from_numpy(rng.uniform(-1, 1, (T, G, H)).astype(np.float32)).bfloat16()
    key = torch.from_numpy(rng.uniform(-0.75, 0.75, (R, 1, 1, D)).astype(np.float32)).bfloat16()
    np.savez(
        ROOT / "inputs.npz",
        q=q.view(torch.uint16).numpy(),
        w=w.view(torch.uint16).numpy(),
        key=key.view(torch.uint16).numpy(),
    )
    dots = q.float().numpy().astype(np.float64).reshape(-1, D) @ key.float().numpy().astype(np.float64).reshape(R, D).T
    np.maximum(dots, 0, out=dots)
    dots = dots.reshape(T, G, H, R)
    dots *= w.float().numpy().astype(np.float64)[..., None]
    scores = dots.sum(axis=2).transpose(1, 0, 2).copy()
    np.save(ROOT / "golden_scores.npy", scores)
    for length in LENGTHS:
        counts = np.clip((length - T + np.arange(T)) // 8, 0, R)
        ids = np.full((G, T, TOPK), -1, dtype=np.int32)
        for group in range(G):
            for token, count in enumerate(counts):
                order = np.argsort(-scores[group, token, :count], kind="stable")[:TOPK]
                ids[group, token, : len(order)] = np.sort(order)
        np.save(ROOT / f"ids_{length}.npy", ids)
    save_json(
        ROOT / "inputs.json",
        {
            "seed": 20946,
            "query": [T, G, H, D],
            "weights": [T, G, H],
            "logical_key": [R, 1, D],
            "paged_key": [R, 1, 1, D],
            "input_dtypes": "all BF16",
            "top_k": TOPK,
            "region_size": 8,
            "head_scale": 1,
            "lengths": list(LENGTHS),
            "block_table": "identity mapping, one request",
            "reference": "FP64 over decoded BF16 input values; ReLU then signed weighted head sum",
            "inputs_sha256": hashlib.sha256((ROOT / "inputs.npz").read_bytes()).hexdigest(),
        },
    )
    print("Generated input bytes and all three FP64 golden cases", flush=True)


def validate(fn, golden, expected_ids, visible, full, destination):
    mask = np.broadcast_to(np.arange(R)[None, None, :] < visible[None, :, None], (G, T, R))
    previous = None
    reports = []
    for repeat in range(3):
        scores, ids = fn()
        torch.npu.synchronize()
        actual_ids = ids.cpu().numpy().copy()
        actual = scores.cpu().numpy().copy() if full else None
        if repeat == 0:
            np.save(destination / "actual_ids.npy", actual_ids)
            if actual is not None:
                np.save(destination / "actual_scores.npy", actual)
        mismatch = np.any(actual_ids != expected_ids, axis=-1)
        entry = {
            "repeat": repeat,
            "id_mismatching_rows": int(mismatch.sum()),
            "id_rows": G * T,
            "ids_exact": bool(not mismatch.any()),
        }
        if actual is not None:
            a, b = actual[mask].astype(np.float64), golden[mask]
            error = np.abs(a - b)
            score_pass = np.isfinite(a) & (error <= 1e-4 + 1e-5 * np.abs(b))
            relative = error / np.maximum(np.abs(b), np.finfo(np.float64).tiny)
            fp32 = golden[mask].astype(np.float32)
            self_consistent = True
            for group in range(G):
                for token, count in enumerate(visible):
                    selected = np.sort(np.argsort(-actual[group, token, :count], kind="stable")[:TOPK])
                    if not np.array_equal(actual_ids[group, token, : len(selected)], selected):
                        self_consistent = False
                        break
            entry.update(
                defined_scores=int(mask.sum()),
                score_failed_count=int((~score_pass).sum()),
                score_max_abs=float(error.max()),
                score_max_relative=float(relative.max()),
                score_rms=float(np.sqrt(np.mean(error * error))),
                scores_within_tolerance=bool(score_pass.all()),
                score_bits_matching_rne_fp64=int(np.sum(actual[mask].view(np.uint32) == fp32.view(np.uint32))),
                ids_match_actual_fp32_scores=self_consistent,
            )
        entry["repeat_bit_identical"] = previous is None or (
            np.array_equal(actual_ids, previous[1])
            and (actual is None or np.array_equal(actual[mask].view(np.uint32), previous[0][mask].view(np.uint32)))
        )
        previous = (actual, actual_ids)
        reports.append(entry)
    passed = all(
        x["ids_exact"]
        and x["repeat_bit_identical"]
        and x.get("scores_within_tolerance", True)
        and x.get("ids_match_actual_fp32_scores", True)
        for x in reports
    )
    return {
        "strict_passed": passed,
        "full_score_precision_verified": full,
        "repeats": reports,
        "tolerance": {"atol": 1e-4, "rtol": 1e-5},
    }


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    events = [torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)]
    wall, stream = [], []
    for _ in range(repeats):
        events[0].record()
        begin = time.perf_counter_ns()
        result = fn()
        events[1].record()
        events[1].synchronize()
        wall.append((time.perf_counter_ns() - begin) / 1000)
        stream.append(events[0].elapsed_time(events[1]) * 1000)
        del result

    def summarize(values):
        return {
            "mean_us": statistics.mean(values),
            "median_us": statistics.median(values),
            "p90_us": float(np.percentile(values, 90)),
            "min_us": min(values),
            "max_us": max(values),
        }

    return {
        "warmup": warmup,
        "repeats": repeats,
        "synchronized_wall": summarize(wall),
        "npu_stream_events": summarize(stream),
        "wall_samples_us": wall,
        "event_samples_us": stream,
        "method": "eager one call per synchronized sample; stream interval may include CPU enqueue gaps",
    }


def main():
    """Share fixed inputs/output storage between accuracy and timing phases.

    Keep both implementations in one harness so the oracle, workload and
    synchronization policy match. Custom/native and ordinary SDK OPP runs
    still use separate processes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--impl", choices=("fused", "small_fp32", "small_bf16", "small_fp32_topk"))
    parser.add_argument("--mode", choices=("full", "ids"), default="full")
    parser.add_argument("--phase", choices=("validate", "time"), default="validate")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.generate:
        generate()
        return
    if args.impl is None:
        parser.error("--impl is required")
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = False
    full = args.mode == "full"
    data = np.load(ROOT / "inputs.npz")
    tensors = [torch.from_numpy(data[name].copy()).view(torch.bfloat16).npu() for name in ("q", "w", "key")]
    q, w, key = tensors
    table = torch.arange(R, dtype=torch.int32).reshape(1, R).npu()
    cu = torch.tensor([0, T], dtype=torch.int32).npu()
    token_positions = torch.arange(T, dtype=torch.int32).npu()
    region_ids = torch.arange(R, dtype=torch.int64).npu()
    length_tensor = torch.empty((1,), dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    if args.impl == "fused":
        torch.ops.load_library(str(ROOT / "bridge.so"))
    golden = np.load(ROOT / "golden_scores.npy", mmap_mode="r")
    for length in LENGTHS:
        destination = ROOT / "results" / f"{args.impl}_{args.mode}_{length}"
        destination.mkdir(parents=True, exist_ok=True)
        length_tensor.copy_(torch.tensor([length], dtype=torch.int32))
        torch.npu.synchronize()
        gc.collect()
        torch.npu.empty_cache()
        input_bytes = torch.npu.memory_allocated()
        if args.impl == "fused":
            scores = torch.empty((G, T, R) if full else (0,), dtype=torch.float32, device="npu")
            ids = torch.empty((G, T, TOPK), dtype=torch.int32, device="npu")
            workspace = torch.empty((0,), dtype=torch.uint8, device="npu")
            required = torch.ops.irs_bench.run(q, w, key, length_tensor, table, cu, scores, ids, workspace, full, False)
            workspace = torch.empty((required,), dtype=torch.uint8, device="npu")

            def operation(scores=scores, ids=ids, workspace=workspace):
                torch.ops.irs_bench.run(q, w, key, length_tensor, table, cu, scores, ids, workspace, full, True)
                return (scores if full else None), ids
        else:
            required = None

            def operation():
                left = q.reshape(-1, D)
                right = key.reshape(R, D).T
                if args.impl.startswith("small_fp32"):
                    left, right = left.float(), right.float()
                dots = torch.mm(left, right).float()
                dots.relu_()
                dots = dots.view(T, G, H, R)
                dots.mul_(w.float()[..., None])
                values = dots.sum(dim=2).permute(1, 0, 2).contiguous()
                visible = torch.div(length_tensor - T + token_positions, 8, rounding_mode="floor").clamp(0, R)
                masked = values.masked_fill(region_ids[None, None, :] >= visible[None, :, None], -float("inf"))
                if args.impl == "small_fp32_topk":
                    _, selected = torch.topk(masked, TOPK, dim=-1, largest=True, sorted=False)
                    selected = selected.to(torch.int32)
                else:
                    _, ranked = torch.sort(masked, dim=-1, descending=True, stable=True)
                    selected = ranked[..., :TOPK]
                selected = torch.where(selected < visible[None, :, None], selected, R)
                selected = selected.sort(dim=-1).values
                output_ids = torch.where(selected < R, selected, -1).to(torch.int32)
                return (values if full else None), output_ids

        visible = np.clip((length - T + np.arange(T)) // 8, 0, R)
        if args.phase == "validate":
            report = validate(operation, golden, np.load(ROOT / f"ids_{length}.npy"), visible, full, destination)
            report.update(
                implementation=args.impl,
                mode=args.mode,
                length=length,
                visible_min=int(visible.min()),
                visible_max=int(visible.max()),
                native_workspace_bytes=required,
                torch_version=torch.__version__,
                torch_npu_version=torch_npu.__version__,
                hardware="910B2C compatibility",
            )
            save_json(destination / "precision.json", report)
            print(
                json.dumps(
                    {
                        "impl": args.impl,
                        "mode": args.mode,
                        "length": length,
                        "strict_passed": report["strict_passed"],
                        "first": report["repeats"][0],
                    }
                ),
                flush=True,
            )
        else:
            report = json.loads((destination / "precision.json").read_text())
            if not all(x["repeat_bit_identical"] for x in report["repeats"]):
                raise RuntimeError("repeatability failed before timing")
            if args.impl == "fused":
                full_precision = json.loads((ROOT / "results" / f"fused_full_{length}" / "precision.json").read_text())
                if not report["strict_passed"] or not full_precision["strict_passed"]:
                    raise RuntimeError("fused accuracy gate failed before timing")
            # Precision diagnostics are completed before timing, and any strict
            # failures stay attached to the measurements for interpretation.
            torch.npu.reset_peak_memory_stats()
            for _ in range(3):
                operation()
            torch.npu.synchronize()
            memory = {
                "resident_input_bytes": input_bytes,
                "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                "extra_peak_above_inputs_bytes": torch.npu.max_memory_allocated() - input_bytes,
                "scope": (
                    "Torch allocator tracked memory including explicit native workspace; "
                    "excludes driver/CANN private pools"
                ),
            }
            result = measure(operation, args.warmup, args.repeats)
            result.update(precision=report, memory=memory)
            save_json(destination / "timing.json", result)
            print(
                json.dumps(
                    {
                        "impl": args.impl,
                        "mode": args.mode,
                        "length": length,
                        "strict_passed": report["strict_passed"],
                        "wall": result["synchronized_wall"],
                        "events": result["npu_stream_events"],
                        "memory": memory,
                    }
                ),
                flush=True,
            )
        if args.impl == "fused":
            del scores, ids, workspace
        # Release default-bound native buffers before measuring the next length.
        del operation
        torch.npu.synchronize()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
