"""Pick the drift FAIL anchor from the pod/anchor.sh ladder, by the rule fixed on 2026-11-11.

    python pod/pick_anchor.py out/   ->  prints the table and writes out/anchor_pick.json

The rule (report.md, "A FAIL anchor that clears the floor"), fixed before the ladder ran:
  1. Target: all-linears. Chosen in advance because quantization, the next variants, changes
     exactly these matrices. all-o-proj and all-down-proj are diagnostics only (was q_proj
     special?) and can never become the anchor.
  2. Ladder: relative noise 1e-3, 3e-3, 1e-2, 3e-2, 1e-1 (x each matrix's std), seed 0.
  3. Anchor = the SMALLEST ladder scale whose mean KL is >= 3 x the 2.7e-3 PASS ceiling
     (8.1e-3) in BOTH decode and prefill mode.
  4. FAIL lines from the anchor, with the v2 rounding rule so the anchor itself reads FAIL:
     KL rounded DOWN to 2 significant figures, top-1 rounded UP to 0.1%, per mode; then the
     lower KL line and the higher top-1 line of the two modes.
  5. If that top-1 line is not strictly below the 98.0% PASS floor, FAIL is KL-only: a top-1
     line at or above the PASS floor would FAIL results that PASS.
  6. If no ladder scale qualifies, there is no anchor. Report it; do not extend the ladder
     without review.
"""

import json
import math
import os
import sys

PASS_KL, PASS_TOP1 = 2.7e-3, 98.0
THRESHOLD_KL = 3 * PASS_KL
ANCHOR_TARGET = "all-linears"
TARGETS = ("all-linears", "all-o-proj", "all-down-proj")
LADDER = ("1e-3", "3e-3", "1e-2", "3e-2", "1e-1")
MODES = ("decode", "prefill")


def round_down_2sf(x):
    e = math.floor(math.log10(x)) - 1
    return math.floor(x / 10**e) * 10**e


def round_up_tenth(x):
    return math.ceil(round(x * 10, 9)) / 10


def load(out_dir):
    rows = {}
    for t in TARGETS:
        for s in LADDER:
            for m in MODES:
                path = os.path.join(out_dir, f"ladder_{t}_{m}_{s}.json")
                if os.path.exists(path):
                    with open(path) as f:
                        d = json.load(f)
                    pert = d.get("variant_meta", {}).get("perturb", {})
                    if pert.get("target") != t or pert.get("scale") != float(s):
                        raise SystemExit(f"{path}: perturb metadata {pert} does not match")
                    rows[(t, s, m)] = (d["mean_kl"], d["top1_agreement_pct"])
    return rows


def pick(rows):
    def qualifies(t, s):
        return all((t, s, m) in rows and rows[(t, s, m)][0] >= THRESHOLD_KL for m in MODES)

    smallest = {t: next((s for s in LADDER if qualifies(t, s)), None) for t in TARGETS}
    missing = [f"{t} {m} {s}" for t in TARGETS for s in LADDER for m in MODES
               if (t, s, m) not in rows]
    scale = smallest[ANCHOR_TARGET]
    result = {"rule": __doc__.split("The rule", 1)[1].strip(), "threshold_kl": THRESHOLD_KL,
              "anchor_target": ANCHOR_TARGET, "missing_runs": missing,
              "smallest_qualifying_scale": smallest, "anchor": None}
    if scale is None:
        return result
    per_mode = {m: {"kl": rows[(ANCHOR_TARGET, scale, m)][0],
                    "top1": rows[(ANCHOR_TARGET, scale, m)][1],
                    "fail_kl": round_down_2sf(rows[(ANCHOR_TARGET, scale, m)][0]),
                    "fail_top1": round_up_tenth(rows[(ANCHOR_TARGET, scale, m)][1])}
                for m in MODES}
    fail_kl = min(v["fail_kl"] for v in per_mode.values())
    fail_top1 = max(v["fail_top1"] for v in per_mode.values())
    kl_only = fail_top1 >= PASS_TOP1
    result["anchor"] = {"target": ANCHOR_TARGET, "scale": float(scale), "per_mode": per_mode,
                        "fail_kl_at_least": fail_kl,
                        "fail_top1_at_most": None if kl_only else fail_top1,
                        "kl_only": kl_only}
    return result


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
    rows = load(out_dir)
    print(f"{'target':14s} {'scale':>6s}  {'decode KL / top-1':>22s}  {'prefill KL / top-1':>22s}"
          f"  >= {THRESHOLD_KL:.1e} both")
    for t in TARGETS:
        for s in LADDER:
            cells = []
            for m in MODES:
                kl, t1 = rows.get((t, s, m), (None, None))
                cells.append(f"{kl:.3e} / {t1:6.2f}%" if kl is not None else "missing")
            ok = all((t, s, m) in rows and rows[(t, s, m)][0] >= THRESHOLD_KL for m in MODES)
            print(f"{t:14s} {s:>6s}  {cells[0]:>22s}  {cells[1]:>22s}  {'yes' if ok else ''}")
    r = pick(rows)
    print()
    if r["missing_runs"]:
        print(f"MISSING {len(r['missing_runs'])} runs: {r['missing_runs']}")
    for t, s in r["smallest_qualifying_scale"].items():
        role = "ANCHOR TARGET" if t == ANCHOR_TARGET else "diagnostic"
        print(f"smallest qualifying scale  {t:14s} {s or 'none':>6s}   ({role})")
    a = r["anchor"]
    if a is None:
        print(f"\nNO ANCHOR: no {ANCHOR_TARGET} scale reached KL >= {THRESHOLD_KL:.1e} in both "
              "modes. Do not extend the ladder without review.")
    else:
        t1 = ("KL-only (top-1 line would not sit below the 98.0% PASS floor)" if a["kl_only"]
              else f"or top-1 <= {a['fail_top1_at_most']}%")
        print(f"\nANCHOR  {a['target']} @ {a['scale']:g}   FAIL: KL >= {a['fail_kl_at_least']:.2g} {t1}")
    with open(os.path.join(out_dir, "anchor_pick.json"), "w") as f:
        json.dump(r, f, indent=2)
    print(f"wrote {os.path.join(out_dir, 'anchor_pick.json')}")


if __name__ == "__main__":
    main()
