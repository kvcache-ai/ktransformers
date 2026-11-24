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
import platform
import subprocess


def check_cpuinfo():
    """Check CPU features via /proc/cpuinfo (Linux only)."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
        return cpuinfo
    except FileNotFoundError:
        return None


def check_cpuinfo_macos():
    """Check CPU features via sysctl (macOS only)."""
    if platform.system() != "Darwin":
        return None

    try:
        # Get CPU brand string
        brand_result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        brand = brand_result.stdout.strip() if brand_result.returncode == 0 else "Unknown"

        # Get CPU features
        features_result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.features"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        features = features_result.stdout.strip().lower().split() if features_result.returncode == 0 else []

        # Get leaf7 features for AVX512
        leaf7_result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.leaf7_features"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        leaf7_features = leaf7_result.stdout.strip().lower().split() if leaf7_result.returncode == 0 else []

        all_features = set(features) | set(leaf7_features)

        return {
            "brand": brand,
            "features": all_features,
        }
    except Exception as e:
        print(f"❌ macOS CPU detection failed: {e}")
        return None


def main():
    print("=" * 70)
    print("KT-Kernel CPU Feature Detection")
    print("=" * 70)
    print()

    # Try Linux first
    cpuinfo = check_cpuinfo()

    # Try macOS if Linux detection failed
    macos_info = None
    if cpuinfo is None:
        macos_info = check_cpuinfo_macos()

    if cpuinfo is None and macos_info is None:
        print("❌ Cannot detect CPU features automatically.")
        print("   Supported platforms: Linux (/proc/cpuinfo), macOS (sysctl)")
        sys.exit(1)

    if cpuinfo is not None:
        # Linux path
        print("Platform: Linux")
        print()

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

    else:
        # macOS path
        print(f"Platform: macOS")
        print(f"CPU Model: {macos_info['brand']}")
        print()

        features = macos_info['features']

        # Check AMX support (not on macOS)
        print("AMX Support (Intel Sapphire Rapids+):")
        amx_flags = ["amx_tile", "amx_int8", "amx_bf16"]
        amx_status = {}
        for flag in amx_flags:
            has_flag = flag in features
            amx_status[flag] = has_flag
            status = "✅" if has_flag else "❌"
            print(f"  {status} {flag.upper()}")

        has_amx = all(amx_status.values())
        print(f"\n  Overall AMX Support: {'✅ YES' if has_amx else '❌ NO (not available on macOS)'}")
        print()

        # Check AVX512 support
        print("AVX512 Support (required for FP8 MoE):")
        avx512_flags = ["avx512f", "avx512_bf16", "avx512_vnni", "avx512_vbmi"]
        avx512_status = {}
        for flag in avx512_flags:
            has_flag = flag in features
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
        has_avx2 = "avx2" in features
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
