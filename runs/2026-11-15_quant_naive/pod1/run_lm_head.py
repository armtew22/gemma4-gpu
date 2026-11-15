"""Run a repo tool with one extra variant registered: int4_naive with lm_head also quantized.

Uses the quantize_lm_head option of variants.patch_quant (658b910) unchanged; this file only
registers a name for it, so no repo code changes. Usage: python run_lm_head.py <tool.py> <args>
"""
import runpy
import sys
from functools import partial

import variants as V

V.VARIANTS["int4_naive_lm_head"] = {
    "builder": V._quant_naive("int4"), "static_cache": False,
    "patch": partial(V.patch_quant, fmt="int4", quantize_lm_head=True),
}
tool = sys.argv[1]
sys.argv = [tool] + sys.argv[2:]
runpy.run_path(tool, run_name="__main__")
