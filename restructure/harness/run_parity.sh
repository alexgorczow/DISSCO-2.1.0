#!/usr/bin/env bash
# Parity harness for the DISSCO viskores-restructuring effort.
#
# Renders a .dissco deterministically (single-thread, fixed seed) with a given
# cmod binary and compares the resulting AIFF against a golden reference.
#
# Usage:
#   run_parity.sh render <cmod-binary> <project.dissco> <out-dir>
#       -> renders project into <out-dir>/SoundFiles, prints the AIFF path + md5
#   run_parity.sh diff <goldenA.aiff> <candidateB.aiff> [max_lsb]
#       -> sample-accurate diff; exits nonzero if max|diff| > max_lsb (default 0)
#
# The single-thread + fixed-seed contract makes the golden bit-exact
# reproducible (see restructure/01_RESTRUCTURE_PLAN.md §3).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cmd="${1:-}"; shift || true
case "$cmd" in
  render)
    cmod="$(realpath "$1")"; proj="$(realpath "$2")"; outdir="$3"
    mkdir -p "$outdir"; outdir="$(realpath "$outdir")"
    cp "$proj" "$outdir/$(basename "$proj")"
    ( cd "$outdir" && echo "1" | "$cmod" "$outdir/$(basename "$proj")" >render.log 2>&1 )
    aiff="$(find "$outdir/SoundFiles" -name '*.aiff' | head -1)"
    echo "rendered: $aiff"
    md5sum "$aiff"
    ;;
  diff)
    a="$1"; b="$2"; maxlsb="${3:-0}"
    python3 "$HERE/aiff_diff.py" "$a" "$b" "$maxlsb"
    ;;
  *)
    echo "usage: run_parity.sh {render|diff} ..." >&2; exit 2;;
esac
