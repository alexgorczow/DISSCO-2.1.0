#!/usr/bin/env python3
"""Compare two AIFF files sample-by-sample. Reports parity metrics.

DISSCO writes 16-bit signed big-endian PCM AIFF (via AuWriter). We parse the
SSND chunk directly rather than rely on the deprecated `aifc` module so this
keeps working on newer Pythons.
"""
import sys, struct
import numpy as np

def read_aiff_int16(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data[0:4] != b'FORM' or data[8:12] not in (b'AIFF', b'AIFC'):
        raise ValueError(f'{path}: not an AIFF file')
    # Walk chunks to find COMM (params) and SSND (samples)
    pos = 12
    nchannels = sampwidth = 0
    nframes = 0
    ssnd = None
    while pos + 8 <= len(data):
        cid = data[pos:pos+4]
        (csize,) = struct.unpack('>I', data[pos+4:pos+8])
        body = data[pos+8:pos+8+csize]
        if cid == b'COMM':
            nchannels, nframes, sampwidth = struct.unpack('>HIH', body[0:8])
        elif cid == b'SSND':
            # first 8 bytes: offset, blocksize
            offset, blocksize = struct.unpack('>II', body[0:8])
            ssnd = body[8+offset:]
        pos += 8 + csize + (csize & 1)  # chunks are word-aligned
    if ssnd is None:
        raise ValueError(f'{path}: no SSND chunk')
    if sampwidth == 16:
        arr = np.frombuffer(ssnd, dtype='>i2').astype(np.int64)
    elif sampwidth == 24:
        nb = (len(ssnd) // 3) * 3
        b3 = np.frombuffer(ssnd[:nb], dtype=np.uint8).reshape(-1, 3).astype(np.int64)
        # big-endian: byte0 is MSB
        val = (b3[:, 0] << 16) | (b3[:, 1] << 8) | b3[:, 2]
        val = np.where(val >= (1 << 23), val - (1 << 24), val)  # sign-extend
        arr = val
    else:
        raise ValueError(f'{path}: unsupported sample width {sampwidth}')
    return arr, nchannels, nframes, sampwidth

def main():
    # optional 3rd arg = max allowed |diff| in LSB for exit-code pass/fail
    thresh = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    a, ca, fa, wa = read_aiff_int16(sys.argv[1])
    b, cb, fb, wb = read_aiff_int16(sys.argv[2])
    fullscale = float(1 << (wa - 1))
    print(f'A: {sys.argv[1]}  channels={ca} frames={fa} width={wa} samples={len(a)}')
    print(f'B: {sys.argv[2]}  channels={cb} frames={fb} width={wb} samples={len(b)}')
    n = min(len(a), len(b))
    if len(a) != len(b):
        print(f'!! length mismatch: {len(a)} vs {len(b)} (comparing first {n})')
    a, b = a[:n], b[:n]
    diff = np.abs(a - b)
    nz = np.count_nonzero(diff)
    print(f'identical samples : {n - nz}/{n} ({100.0*(n-nz)/n:.4f}%)')
    print(f'differing samples : {nz} ({100.0*nz/n:.4f}%)')
    print(f'max |diff|        : {diff.max()} LSB (full-scale={int(fullscale)})')
    print(f'mean |diff|       : {diff.mean():.6f} LSB')
    if diff.max() > 0:
        # RMS in dBFS
        rms = np.sqrt(np.mean((a - b).astype(np.float64)**2))
        rms_dbfs = 20*np.log10(rms/fullscale) if rms > 0 else -999
        print(f'RMS error         : {rms:.6f} LSB  ({rms_dbfs:.2f} dBFS)')
        # bytes where first difference occurs
        first = int(np.argmax(diff > 0))
        print(f'first diff at sample #{first} (frame {first//max(ca,1)}): A={a[first]} B={b[first]}')
    dmax = int(diff.max()) if len(diff) else 0
    print(f'threshold         : {thresh} LSB  ->  {"PASS" if dmax <= thresh else "FAIL"}')
    sys.exit(0 if (len(a) == len(b) and dmax <= thresh) else 1)

if __name__ == '__main__':
    main()
