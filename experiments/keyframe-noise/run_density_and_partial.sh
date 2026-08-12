#!/usr/bin/env bash
# Sizes the keyframe proposer's job: how many keyframes does it have to emit,
# and does it have to be accurate about every joint?
#
#   A. density   -- how does keyframe error scale with how sparsely you specify?
#   B. partial   -- does error injected into one body part stay there, and does
#                   leaving joints unspecified reduce the damage?
#
# ~1.5 h on a 9070 XT. Writes results/dens-*.csv and results/part-*.csv.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."

PY=".venv-rocm/bin/python"
export PYTORCH_HIP_ALLOC_CONF=roundup_power2_divisions:16
COMMON="--num_samples 16 --seed 0 --thetas 0,10,30 --tols 1.0,0.0"

# A. density. condmdi_randomframes, all joints observed -- identical protocol to
# the committed sweep, so tl=10 reproduces it and the other rows are comparable.
for tl in 5 10 20 40; do
    echo "=== density: trans_length=$tl ==="
    $PY -u experiments/keyframe-noise/run_sweep.py $COMMON \
        --trans_length "$tl" --tag "dens-tl$tl"
done

# B. partial pose. condmdi_randomframejoints was trained with per-joint keyframe
# masks; randomframes only ever saw whole-frame keys, so a joint subset would be
# out of distribution for it. Config 1 is the on-this-checkpoint baseline that
# makes 2 and 3 readable.
run_part () {  # obs perturb tag
    echo "=== partial: obs=$1 perturb=$2 ==="
    $PY -u experiments/keyframe-noise/run_sweep.py $COMMON \
        --model condmdi_randomframejoints --trans_length 10 \
        --obs_joints "$1" --perturb_joints "$2" --tag "$3"
}
run_part all   all   part-obsall-pall
run_part all   arms  part-obsall-parms
run_part upper arms  part-obsupper-parms
run_part lower legs  part-obslower-plegs

echo "done"
