#!/usr/bin/env python3
"""Align safetensors header to 512-byte boundary for MESH O_DIRECT."""
import struct
import os
import sys
import shutil


def align_safetensors(src, dst):
    with open(src, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_bytes = f.read(header_len)
        # Read rest in chunks to avoid loading 1.5G into memory
        data = f.read()

    # Pad header with spaces so data_start = 8 + padded_len is 512-aligned
    target = ((header_len + 8 + 511) // 512) * 512
    padded_len = target - 8
    padding = padded_len - header_len
    padded_header = header_bytes + b' ' * padding

    with open(dst, 'wb') as f:
        f.write(struct.pack('<Q', padded_len))
        f.write(padded_header)
        f.write(data)


def main():
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    os.makedirs(dst_dir, exist_ok=True)

    # Copy non-safetensors files (config.json, etc.)
    for f in os.listdir(src_dir):
        if not f.endswith('.safetensors') and not f.endswith('.tmp'):
            src_f = os.path.join(src_dir, f)
            dst_f = os.path.join(dst_dir, f)
            if os.path.isfile(src_f):
                shutil.copy2(src_f, dst_f)

    # Align safetensors files
    files = sorted([f for f in os.listdir(src_dir)
                    if f.endswith('.safetensors') and not f.endswith('.tmp')])

    for i, f in enumerate(files):
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        align_safetensors(src, dst)
        # Verify alignment
        with open(dst, 'rb') as fp:
            hl = struct.unpack('<Q', fp.read(8))[0]
            aligned = (hl + 8) % 512 == 0
        size_gib = os.path.getsize(dst) / 1024 ** 3
        print(f'[{i+1}/{len(files)}] {f} -> {size_gib:.2f} GiB, aligned={aligned}')
        sys.stdout.flush()

    print('All done!')


if __name__ == '__main__':
    main()
