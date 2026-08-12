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
RESULTS="experiments/keyframe-noise/results"

# Resumable: each config takes ~9 min, so a config that already has a CSV is
# skipped rather than regenerated. Delete its CSV to force a re-run.
run () {  # tag, then run_sweep args
    local tag="$1"; shift
    if compgen -G "$RESULTS/$tag-*.csv" > /dev/null; then
        echo "=== skip $tag (already have $(ls -1 "$RESULTS/$tag-"*.csv | tail -1)) ==="
        return
    fi
    echo "=== $tag ==="
    $PY -u experiments/keyframe-noise/run_sweep.py $COMMON --tag "$tag" "$@"
}

# A. density. condmdi_randomframes, all joints observed -- identical protocol to
# the committed sweep, so tl=10 reproduces it and the other rows are comparable.
for tl in 5 10 20 40; do
    run "dens-tl$tl" --trans_length "$tl"
done

# B. partial pose. condmdi_randomframejoints was trained with per-joint keyframe
# masks; randomframes only ever saw whole-frame keys, so a joint subset would be
# out of distribution for it. Config 1 is the on-this-checkpoint baseline that
# makes 2 and 3 readable.
run_part () {  # obs perturb tag
    run "$3" --model condmdi_randomframejoints --trans_length 10 \
        --obs_joints "$1" --perturb_joints "$2"
}
run_part all   all   part-obsall-pall
run_part all   arms  part-obsall-parms
run_part upper arms  part-obsupper-parms
run_part lower legs  part-obslower-plegs

echo "done"
