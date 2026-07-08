#!/usr/bin/env python3
"""Tail-and-play client for DISSCO's streaming preview (LASS_STREAM).

Stream format: 12-byte header ("DSTR" magic, u32 sample rate, u32 channels),
then interleaved float32 frames appended as time windows finalize
(restructure/10_REALTIME_LISTENING.md).

Usage:
  # measurement only (works everywhere, incl. headless WSL):
  python3 listen.py /tmp/piece.pcm

  # with live playback if the `sounddevice` package + an audio device exist:
  python3 listen.py /tmp/piece.pcm --play

  # typical producer, run in another shell (or via --run):
  LASS_COMPOSITE=det LASS_PIPELINE=gpu-fast LASS_STREAM=/tmp/piece.pcm \
      cmod piece.dissco

Reports time-to-first-audio and the sustained real-time margin (rendered
seconds per wall second). In Jupyter, use `tail_windows()` and feed each
chunk to IPython.display.Audio(..., autoplay=True) like the DISSCO_hpc
notebooks do.
"""
import argparse
import os
import struct
import subprocess
import sys
import time

HDR = 12


def wait_for_header(path, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if os.path.getsize(path) >= HDR:
                with open(path, "rb") as f:
                    magic, rate, ch = struct.unpack("<III", f.read(HDR))
                if magic != 0x52545344:
                    sys.exit(f"not a DISSCO stream (magic {magic:#x})")
                return rate, ch
        except FileNotFoundError:
            pass
        time.sleep(0.05)
    sys.exit("timed out waiting for stream header")


def tail_windows(path, rate, ch, poll=0.05, idle_timeout=30.0):
    """Yield (t_wall, frames_bytes) as new audio appears; ends on idle."""
    pos = HDR
    frame = 4 * ch
    last_new = time.time()
    while True:
        size = os.path.getsize(path)
        avail = (size - pos) // frame * frame
        if avail > 0:
            with open(path, "rb") as f:
                f.seek(pos)
                data = f.read(avail)
            pos += len(data)
            last_new = time.time()
            yield time.time(), data
        else:
            if time.time() - last_new > idle_timeout:
                return
            time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    ap.add_argument("--play", action="store_true", help="play via sounddevice")
    ap.add_argument("--run", help="launch this cmod command first (shell)")
    args = ap.parse_args()

    proc = None
    t_launch = time.time()
    if args.run:
        proc = subprocess.Popen(args.run, shell=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.PIPE)
        proc.stdin.write(b"1\n")
        proc.stdin.flush()

    rate, ch = wait_for_header(args.stream)
    print(f"stream: {rate} Hz, {ch} ch")

    player = None
    if args.play:
        try:
            import numpy as np
            import sounddevice as sd
            player = sd.OutputStream(samplerate=rate, channels=ch,
                                     dtype="float32")
            player.start()
        except Exception as e:
            print(f"(playback unavailable: {e}; measuring only)")
            player = None

    total_frames = 0
    t_first = None
    for t_wall, data in tail_windows(args.stream, rate, ch):
        if t_first is None:
            t_first = t_wall
            print(f"time-to-first-audio: {t_first - t_launch:.2f}s "
                  f"(from client start)")
        total_frames += len(data) // (4 * ch)
        rendered_s = total_frames / rate
        elapsed = t_wall - t_launch  # honest: includes composition+startup
        margin = rendered_s / elapsed if elapsed > 0.2 else float("inf")
        print(f"\r{rendered_s:8.2f}s rendered | wall {elapsed:7.2f}s since "
              f"launch | margin {margin:5.2f}x realtime", end="")
        if player is not None:
            import numpy as np
            player.write(np.frombuffer(data, dtype="<f4").reshape(-1, ch))
    print()
    if proc is not None:
        proc.wait()
    print(f"done: {total_frames / rate:.2f}s of audio")


if __name__ == "__main__":
    main()
