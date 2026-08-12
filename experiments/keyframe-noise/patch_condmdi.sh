#!/usr/bin/env bash
# Make the 2023-era CondMDI checkout run on modern numpy / torch / ROCm.
#
# CondMDI pins numpy==1.21.5 and torch==1.13.1. We run numpy 2.x and
# torch 2.13.0+rocm7.2, so the deprecated scalar aliases removed in numpy 1.24
# have to go. Every substitution here is semantics-preserving: np.float was
# always an alias for the builtin float, np.int for int, and so on.
#
# Idempotent -- safe to re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # resolve before we cd away
REPO="${1:-/var/home/akhil/live-animation/third_party/CondMDI}"
cd "$REPO"

echo "== patching deprecated numpy aliases under $REPO =="

# Word-boundary matches only, so np.float64 / np.int32 / np.bool_ are untouched.
find . -name '*.py' -not -path './.git/*' -print0 | xargs -0 sed -i \
  -e 's/\bnp\.float\b/float/g' \
  -e 's/\bnp\.int\b/int/g' \
  -e 's/\bnp\.bool\b/bool/g' \
  -e 's/\bnp\.object\b/object/g' \
  -e 's/\bnp\.str\b/str/g' \
  -e 's/\bnp\.complex\b/complex/g'

echo "-- remaining bare aliases (should be none):"
grep -rn --include='*.py' -E '\bnp\.(float|int|bool|object|str|complex)\b' . || echo "   none"

# numpy 2 removed np.NaN / np.Inf in favour of np.nan / np.inf.
find . -name '*.py' -not -path './.git/*' -print0 | xargs -0 sed -i \
  -e 's/\bnp\.NaN\b/np.nan/g' \
  -e 's/\bnp\.Inf\b/np.inf/g' \
  -e 's/\bnp\.NAN\b/np.nan/g'

# numpy 2 removed the `in1d` alias.
find . -name '*.py' -not -path './.git/*' -print0 | xargs -0 sed -i \
  -e 's/\bnp\.in1d\b/np.isin/g'

# torch.load defaults to weights_only=True from torch 2.6; these checkpoints are
# full pickles authored by the paper's own training script, so opt in explicitly
# rather than globally trusting every load.
if ! grep -q "weights_only=False" utils/model_util.py 2>/dev/null; then
  sed -i 's/torch\.load(\(.*\))/torch.load(\1, weights_only=False)/g' utils/model_util.py || true
fi

# Semantic changes the experiment depends on, kept as a real patch so they are
# reviewable rather than buried in sed:
#   - a keyframe_tolerance knob on the imputation override (the §8d.9 axis)
#   - lazy SMPL construction, so sampling does not drag in chumpy
PATCH="$SCRIPT_DIR/condmdi-tolerance.patch"
if [ -f "$PATCH" ]; then
  if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo "-- tolerance patch already applied"
  elif git apply --check "$PATCH" >/dev/null 2>&1; then
    git apply "$PATCH"
    echo "-- applied tolerance patch"
  else
    echo "!! tolerance patch does not apply cleanly -- apply $PATCH by hand" >&2
    exit 1
  fi
else
  echo "!! $PATCH not found" >&2
  exit 1
fi

echo "== done =="
