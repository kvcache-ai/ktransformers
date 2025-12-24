#!/usr/bin/env python3
"""
CPU feature detection script for kt-kernel.

This script checks if your CPU supports the required instruction sets for FP8 MoE:
- AVX512F (foundation)
- AVX512_BF16 (BF16 dot product)
- AVX512_VNNI (VNNI instructions)
- AVX512_VBMI (byte permutation)

Usage:
    python3 scripts/check_cpu_features.py
"""

import os
import sys


def check_cpuinfo():
    """Check CPU features via /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
        return cpuinfo
    except FileNotFoundError:
        return None


def main():
    print("=" * 70)
    print("KT-Kernel CPU Feature Detection")
    print("=" * 70)
    print()

    cpuinfo = check_cpuinfo()

    if cpuinfo is None:
        print("❌ /proc/cpuinfo not found (not on Linux?)")
        print("   Cannot detect CPU features automatically.")
        sys.exit(1)

    # Extract CPU model
    for line in cpuinfo.split("\n"):
        if "model name" in line:
            model = line.split(":")[1].strip()
            print(f"CPU Model: {model}")
            break
    print()

    # Check AMX support
    print("AMX Support (Intel Sapphire Rapids+):")
    amx_flags = ["amx_tile", "amx_int8", "amx_bf16"]
    amx_status = {}
    for flag in amx_flags:
        has_flag = flag in cpuinfo
        amx_status[flag] = has_flag
        status = "✅" if has_flag else "❌"
        print(f"  {status} {flag.upper()}")

    has_amx = all(amx_status.values())
    print(f"\n  Overall AMX Support: {'✅ YES' if has_amx else '❌ NO'}")
    print()

    # Check AVX512 support
    print("AVX512 Support (required for FP8 MoE):")
    avx512_flags = ["avx512f", "avx512_bf16", "avx512_vnni", "avx512_vbmi"]
    avx512_status = {}
    for flag in avx512_flags:
        has_flag = flag in cpuinfo
        avx512_status[flag] = has_flag
        status = "✅" if has_flag else "❌"
        flag_desc = {
            "avx512f": "AVX512F (foundation)",
            "avx512_bf16": "AVX512_BF16 (BF16 dot product)",
            "avx512_vnni": "AVX512_VNNI (VNNI instructions)",
            "avx512_vbmi": "AVX512_VBMI (byte permutation)",
        }
        print(f"  {status} {flag_desc.get(flag, flag.upper())}")

    has_avx512_full = all(avx512_status.values())
    print(f"\n  Overall AVX512 Support: {'✅ YES' if has_avx512_full else '❌ NO'}")

    if not has_avx512_full and avx512_status["avx512f"]:
        missing = [f for f in avx512_flags if not avx512_status[f]]
        print(f"  ⚠️  Warning: AVX512F detected but missing: {', '.join(missing)}")
        print(f"      kt-kernel will fall back to AVX2 mode")
    print()

    # Check AVX2 support
    print("AVX2 Support (fallback):")
    has_avx2 = "avx2" in cpuinfo
    status = "✅" if has_avx2 else "❌"
    print(f"  {status} AVX2")
    print()

    # Recommendation
    print("=" * 70)
    print("Recommendation:")
    print("=" * 70)
    if has_amx:
        print("✅ Your CPU supports AMX - you can use the highest performance mode!")
        print("   Build with: -DKTRANSFORMERS_CPU_USE_AMX_AVX512=ON -DKTRANSFORMERS_CPU_USE_AMX=ON")
    elif has_avx512_full:
        print("✅ Your CPU supports full AVX512 (F/BF16/VNNI/VBMI) - FP8 MoE will work!")
        print("   Build with: -DKTRANSFORMERS_CPU_USE_AMX_AVX512=ON")
    elif avx512_status.get("avx512f", False):
        print("⚠️  Your CPU has AVX512F but missing required extensions.")
        print("   FP8 MoE will NOT work. kt-kernel will fall back to AVX2 mode.")
        print("   Missing extensions:", ", ".join([f for f in avx512_flags if not avx512_status.get(f, False)]))
    elif has_avx2:
        print("ℹ️  Your CPU supports AVX2 only - basic compatibility mode.")
        print("   FP8 MoE will NOT be available, but other features will work.")
    else:
        print("❌ Your CPU does not support the minimum required instruction set (AVX2).")
        print("   kt-kernel may not work on this system.")
    print()


if __name__ == "__main__":
    main()
