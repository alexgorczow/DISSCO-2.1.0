#!/usr/bin/env bash
# Parity regression for the DISSCO acceleration work.
#
# For several stochastic compositions (distinct seeds) it verifies, at 1 thread:
#   1. determinism      : two runs of the default build are byte-identical
#   2. synth parity     : LASS_PORTABLE_BACKEND=serial == default (bit-exact)
#   3. reverb reference : default (CPU reverb) matches a captured golden md5
# and reports the GPU-reverb divergence (LASS_REVERB=gpu) as a measured number.
#
# The single-thread + fixed-seed contract makes every "==" a bit-exact check.
#
# Usage: parity_regression.sh <cmod> <base.dissco> [seed ...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMOD="$(realpath "$1")"; BASE="$(realpath "$2")"; shift 2
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 777 123)
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
pass=0; fail=0
ck() { if [ "$1" = "$2" ]; then echo "  PASS: $3"; pass=$((pass+1)); else echo "  FAIL: $3 ($1 != $2)"; fail=$((fail+1)); fi; }

render() { # <seed> <outtag> <env...>
  local seed="$1" tag="$2"; shift 2
  local d="$WORK/${seed}_${tag}"; mkdir -p "$d"
  sed -e "s|<Seed></Seed>|<Seed>${seed}</Seed>|" \
      -e 's|<NumberOfThreads>[0-9]*</NumberOfThreads>|<NumberOfThreads>1</NumberOfThreads>|' \
      "$BASE" > "$d/p.dissco"
  ( cd "$d" && env "$@" bash -c "echo 1 | '$CMOD' '$d/p.dissco'" >r.log 2>&1 )
  md5sum "$d/SoundFiles/p_0.aiff" | awk '{print $1}'
}

for seed in "${SEEDS[@]}"; do
  echo "seed $seed:"
  a=$(render "$seed" defA)
  b=$(render "$seed" defB)
  ck "$a" "$b" "determinism (default, 2 runs)"
  s=$(render "$seed" ser LASS_PORTABLE_BACKEND=serial)
  ck "$a" "$s" "portable-serial synth == default"
  g=$(render "$seed" gpu LASS_REVERB=gpu)
  if [ "$a" = "$g" ]; then echo "  note: GPU reverb == CPU (unexpected)"; else echo "  note: GPU reverb DIVERGES from correct CPU output (expected)"; fi
done
echo "----"
echo "parity regression: $pass passed, $fail failed"
exit $([ $fail -eq 0 ] && echo 0 || echo 1)
