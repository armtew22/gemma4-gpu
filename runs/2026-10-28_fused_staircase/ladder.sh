cd /root
for s in 3e-3 1e-2; do
  /opt/venv/bin/python eval_drift.py --mode decode --variant eager --perturb $s --perturb-target all-q-proj --reference ref_decode_eager.pt --out out/ladder_decode_$s.json > out/ladder_decode_$s.log 2>&1
  /opt/venv/bin/python eval_drift.py --variant eager --perturb $s --perturb-target all-q-proj --reference ref_50k_bos.pt --out out/ladder_prefill_$s.json > out/ladder_prefill_$s.log 2>&1
done
touch out/LADDER_DONE
