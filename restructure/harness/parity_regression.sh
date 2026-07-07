#!/usr/bin/env bash
# Parity regression for the DISSCO acceleration work.
#
# For several stochastic compositions (distinct seeds) it verifies:
#   1. determinism      : two runs of the default build are byte-identical (1t)
#   2. synth parity     : LASS_PORTABLE_BACKEND=serial == default (bit-exact, 1t)
#   3. det composite    : LASS_COMPOSITE=det gives the SAME md5 at 1 and 8
#                         threads (thread-count-independent output)
#   4. det-gpu parity   : LASS_COMPOSITE=det-gpu == det (GPU adds bit-identical
#                         to CPU adds; on non-CUDA builds det-gpu falls back to
#                         det, so the check passes trivially)
# and reports the GPU-reverb divergence (LASS_REVERB=gpu) as a measured number.
#
# Checks 1-2 rely on the single-thread + fixed-seed contract; checks 3-4 are
# the deterministic-composite contract (restructure/06_DETERMINISTIC_COMPOSITE.md).
# Golden md5s for known projects live in goldens/MANIFEST.md.
#
# Usage: parity_regression.sh <cmod> <base.dissco> [seed ...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMOD="$(realpath "$1")"; BASE="$(realpath "$2")"; shift 2
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 777 123)
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
pass=0; fail=0
ck() { if [ "$1" = "$2" ]; then echo "  PASS: $3"; pass=$((pass+1)); else echo "  FAIL: $3 ($1 != $2)"; fail=$((fail+1)); fi; }

render() { # <seed> <threads> <outtag> <env...>
  local seed="$1" threads="$2" tag="$3"; shift 3
  local d="$WORK/${seed}_${tag}"; mkdir -p "$d"
  sed -e "s|<Seed></Seed>|<Seed>${seed}</Seed>|" \
      -e "s|<NumberOfThreads>[0-9]*</NumberOfThreads>|<NumberOfThreads>${threads}</NumberOfThreads>|" \
      "$BASE" > "$d/p.dissco"
  ( cd "$d" && env "$@" bash -c "echo 1 | '$CMOD' '$d/p.dissco'" >r.log 2>&1 )
  md5sum "$d/SoundFiles/p_0.aiff" | awk '{print $1}'
}

for seed in "${SEEDS[@]}"; do
  echo "seed $seed:"
  a=$(render "$seed" 1 defA)
  b=$(render "$seed" 1 defB)
  ck "$a" "$b" "determinism (default, 2 runs)"
  s=$(render "$seed" 1 ser LASS_PORTABLE_BACKEND=serial)
  ck "$a" "$s" "portable-serial synth == default"
  d1=$(render "$seed" 1 det1 LASS_COMPOSITE=det)
  d8=$(render "$seed" 8 det8 LASS_COMPOSITE=det)
  ck "$d1" "$d8" "det composite: 1t == 8t (thread-count independent)"
  dg=$(render "$seed" 8 detgpu LASS_COMPOSITE=det-gpu)
  ck "$d1" "$dg" "det-gpu == det (cross-device)"
  g=$(render "$seed" 1 gpu LASS_REVERB=gpu)
  if [ "$a" = "$g" ]; then echo "  note: GPU reverb == CPU (unexpected)"; else echo "  note: GPU reverb DIVERGES from correct CPU output (expected)"; fi
done
echo "----"
echo "parity regression: $pass passed, $fail failed"
exit $([ $fail -eq 0 ] && echo 0 || echo 1)
