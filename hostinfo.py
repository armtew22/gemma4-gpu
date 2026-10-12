"""Host identity for every results file.

Two pods both labelled "NVIDIA L4" differed 1.76x in decode throughput (17.6 vs 32.4
tok/s) because decode launches ~3000 CUDA kernels per step and the host CPU gates how
fast they are issued. Absolute throughput is therefore meaningless without knowing which
machine produced it, so every JSON this repo writes records the host.
"""

import os
import platform
import subprocess


def _cpu_model():
    try:
        if platform.system() == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip() or "unknown"
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def _nvidia_smi(query):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return out.splitlines()[0].strip() if out else "unknown"
    except Exception:
        return "unknown"


def host_info():
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "vcpu_count": os.cpu_count(),
        "gpu_name": _nvidia_smi("name"),
        "gpu_memory": _nvidia_smi("memory.total"),
        "gpu_driver": _nvidia_smi("driver_version"),
        "gpu_sm_clock_max": _nvidia_smi("clocks.max.sm"),
        "gpu_power_limit": _nvidia_smi("power.limit"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["torch_gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


def print_host(info=None):
    info = info or host_info()
    print(f"host   {info['cpu_model']}  ({info['vcpu_count']} vCPU)")
    print(f"gpu    {info['gpu_name']}  {info['gpu_memory']}  driver {info['gpu_driver']}")
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(host_info(), indent=2))
