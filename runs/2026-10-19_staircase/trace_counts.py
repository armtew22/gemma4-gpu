"""Ad-hoc, read-only: per-decode-step GPU kernels and CPU launch calls from a profile trace.

Uses the same gen_fwd_* windows and correlation-id attribution as profile_decode.py, and
additionally counts the CPU-side launch API calls inside each window, split by kind:
runtime cudaLaunchKernel*, driver cuLaunchKernel* (how Triton/inductor kernels launch),
and graph launches (cudaGraphLaunch / cuGraphLaunch).
"""
import json
import statistics
import sys
from collections import defaultdict

path, skip = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2
events = json.load(open(path))["traceEvents"]

windows, kernels_by_corr, launches = {}, defaultdict(list), []
for e in events:
    cat, name = e.get("cat"), str(e.get("name", ""))
    if cat == "user_annotation" and name.startswith("gen_fwd_"):
        windows[name] = (e["ts"], e["ts"] + e.get("dur", 0))
    elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        corr = e.get("args", {}).get("correlation")
        if corr is not None:
            kernels_by_corr[corr].append(e)
    elif cat in ("cuda_runtime", "cuda_driver", "runtime"):
        launches.append((e["ts"], name, e.get("args", {}).get("correlation")))

if not windows:
    print(json.dumps({"trace": path, "error": "no gen_fwd_* annotations"}))
    sys.exit(3)


def kind(name):
    if "GraphLaunch" in name:
        return "graph_launch"
    if name.startswith("cudaLaunchKernel"):
        return "cudaLaunchKernel"
    if name.startswith("cuLaunchKernel"):
        return "cuLaunchKernel"
    return None


names = sorted(windows)
decode = names[1:][skip:]
per = defaultdict(list)
for w in decode:
    lo, hi = windows[w]
    counts, gpu_k, gpu_mem, gpu_us = defaultdict(int), 0, 0, 0.0
    for ts, name, corr in launches:
        if lo <= ts <= hi:
            k = kind(name)
            if k:
                counts[k] += 1
            for ev in kernels_by_corr.get(corr, ()):
                gpu_us += ev.get("dur", 0)
                if ev.get("cat") == "kernel":
                    gpu_k += 1
                else:
                    gpu_mem += 1
    for k in ("cudaLaunchKernel", "cuLaunchKernel", "graph_launch"):
        per[k].append(counts[k])
    per["gpu_kernels"].append(gpu_k)
    per["gpu_memcpy_memset"].append(gpu_mem)
    per["gpu_ms"].append(gpu_us / 1000)

out = {"trace": path, "decode_steps_found": len(names) - 1, "decode_steps_analyzed": len(decode)}
for k, v in per.items():
    out[k + "_per_step"] = statistics.mean(v) if v else 0
out["cpu_launch_calls_per_step"] = (
    out["cudaLaunchKernel_per_step"] + out["cuLaunchKernel_per_step"] + out["graph_launch_per_step"]
)
print(json.dumps(out, indent=2))
