#!/bin/bash
# Full WHO unlearning pipeline: unlearn (3 methods) -> oracle -> audit -> attacks.
# Assumes checkpoints/trained_model.pt exists (EPOCHS=15 python train.py).
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
V=.venv/bin/python

echo "### 1/11 Gradient Ascent";        $V unlearn_ga.py
echo "### 2/11 Gradient Difference";    $V unlearn_gd.py
echo "### 3/11 NPO";                    $V unlearn_npo.py
echo "### 4/11 Oracle (retrain)";       EPOCHS=15 $V oracle.py
echo "### 5/11 Audit GA";               UNLEARNED=unlearned_model.pt $V audit.py
echo "### 6/11 Audit GD";               UNLEARNED=unlearned_gd.pt    $V audit.py
echo "### 7/11 Audit NPO";              UNLEARNED=unlearned_npo.pt   $V audit.py
echo "### 8/11 Comparison";             $V compare.py
echo "### 9/11 Membership inference";   $V attack_mia.py
echo "### 10/11 Relearning attack";     $V attack_relearn.py
echo "### 11/11 Sequential + sweep";    $V sequential.py && $V sweep.py
echo "### DONE"
