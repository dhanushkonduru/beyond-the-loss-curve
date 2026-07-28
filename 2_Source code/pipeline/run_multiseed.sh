#!/bin/bash
# Multi-seed replication: snapshot the existing seed-42 run, then repeat the full
# pipeline for further seeds so audit/attack metrics can be reported as mean +/- std.
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
V=.venv/bin/python
mkdir -p results/seeds

snapshot () {   # $1 = seed
  for m in unlearned_model unlearned_gd unlearned_npo; do
    cp results/audit_matrix_$m.json results/seeds/seed$1_$m.json
  done
  cp results/mia.json     results/seeds/seed$1_mia.json
  cp results/relearn.json results/seeds/seed$1_relearn.json
}

echo "### snapshot seed 42 (already computed)"
snapshot 42

for S in 43 44; do
  echo "### ===== SEED $S : train ====="
  SEED=$S EPOCHS=15 $V train.py
  echo "### seed $S : unlearn (GA, GD, NPO)"
  SEED=$S $V unlearn_ga.py
  SEED=$S $V unlearn_gd.py
  SEED=$S $V unlearn_npo.py
  echo "### seed $S : oracle"
  SEED=$S EPOCHS=15 $V oracle.py
  echo "### seed $S : audits"
  UNLEARNED=unlearned_model.pt $V audit.py
  UNLEARNED=unlearned_gd.pt    $V audit.py
  UNLEARNED=unlearned_npo.pt   $V audit.py
  echo "### seed $S : attacks"
  $V attack_mia.py
  $V attack_relearn.py
  snapshot $S
  echo "### ===== SEED $S DONE ====="
done
echo "### ALL SEEDS DONE"
