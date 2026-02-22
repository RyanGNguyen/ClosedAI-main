#!/usr/bin/env python3
"""
Profile three attention kernel versions using both nsys and ncu
Versions:
1. kernels/attention.cuh (Flash Attention with Tensor Cores)
2. kernels_req_4/attention_no_tc.cuh (Flash Attention without Tensor Cores)
3. kernels_req_3/attention.cuh (Classic Attention with cuBLAS)
"""

import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime

# Kernel configurations
KERNELS = {
    'flash_tc': {
        'name': 'Flash Attention (Tensor Cores)',
        'kernel_dir': 'kernels',
        'kernel_file': '../kernels/attention.cuh',
        'define': 'PROFILE_FLASH_TC'
    },
    'flash_no_tc': {
        'name': 'Flash Attention (No Tensor Cores)',
        'kernel_dir': 'kernels_req_4',
        'kernel_file': '../kernels_req_4/attention_no_tc.cuh',
        'define': 'PROFILE_FLASH_NO_TC'
    },
    'classic': {
        'name': 'Classic Attention (cuBLAS)',
        'kernel_dir': 'kernels_req_3',
        'kernel_file': '../kernels_req_3/attention.cuh',
        'define': 'PROFILE_CLASSIC'
    }
}

NVCC = 'nvcc'
CFLAGS = '-O3 -arch=sm_86 -std=c++11 -rdc=true -g -lineinfo'
CUBLAS_INCLUDES = '-I/usr/local/cuda/include'
CUBLAS_LIBS = '-L/usr/local/cuda/lib64 -lcublas -lcublasLt -lnvToolsExt'

BENCHMARK_SCRIPT = 'kernels_req_4/benchmark_attention_multi.cu'
OUTPUT_DIR = 'profiling_results'

def run_command(cmd, verbose=True):
    """Execute a shell command and return success status"""
    if verbose:
        print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def build_benchmark(kernel_key, kernel_info):
    """Build benchmark binary for a specific kernel version"""
    print(f"\n{'='*70}")
    print(f"Building: {kernel_info['name']}")
    print(f"{'='*70}")

    binary_name = f"benchmark_{kernel_key}"
    object_file = f"benchmark_{kernel_key}.o"

    # Compile
    compile_cmd = (
        f"{NVCC} {CFLAGS} {CUBLAS_INCLUDES} "
        f"-D{kernel_info['define']} "
        f"-I. -c {BENCHMARK_SCRIPT} -o {object_file}"
    )

    if not run_command(compile_cmd):
        print(f"ERROR: Compilation failed for {kernel_key}")
        return None

    # Link
    link_cmd = f"{NVCC} {CFLAGS} -o {binary_name} {object_file} {CUBLAS_LIBS}"
    if not run_command(link_cmd):
        print(f"ERROR: Linking failed for {kernel_key}")
        return None

    print(f"✓ Built: {binary_name}")
    return binary_name

def run_nsys_profile(binary_name, kernel_key, kernel_info):
    """Run nsys profiling on a benchmark binary"""
    print(f"\n{'='*70}")
    print(f"Running nsys for: {kernel_info['name']}")
    print(f"{'='*70}")

    output_file = f"{OUTPUT_DIR}/nsys_{kernel_key}.nsys-rep"
    cmd = f"nsys profile -o {output_file} -f true ./{binary_name}"

    if run_command(cmd):
        print(f"✓ nsys results saved to: {output_file}")
        return output_file
    else:
        print(f"ERROR: nsys profiling failed for {kernel_key}")
        return None

def run_ncu_profile(binary_name, kernel_key, kernel_info):
    """Run ncu profiling on a benchmark binary"""
    print(f"\n{'='*70}")
    print(f"Running ncu for: {kernel_info['name']}")
    print(f"{'='*70}")

    output_file = f"{OUTPUT_DIR}/ncu_{kernel_key}.ncu-rep"
    cmd = f"ncu --set full -o {output_file} ./{binary_name}"

    if run_command(cmd):
        print(f"✓ ncu results saved to: {output_file}")
        return output_file
    else:
        print(f"ERROR: ncu profiling failed for {kernel_key}")
        return None

def export_ncu_results(ncu_file, kernel_key):
    """Export ncu results to CSV for analysis"""
    csv_file = f"{OUTPUT_DIR}/ncu_{kernel_key}.csv"
    cmd = f"sudo ncu --import {ncu_file} --csv -o {csv_file}"

    if run_command(cmd, verbose=False):
        print(f"  ✓ Exported to CSV: {csv_file}")
        return csv_file
    return None

def export_nsys_results(nsys_file, kernel_key):
    """Export nsys results to TXT for execution time analysis"""
    # First, export nsys-rep to sqlite
    sqlite_file = f"{OUTPUT_DIR}/nsys_{kernel_key}.sqlite"
    export_cmd = f"nsys export --type sqlite --output {sqlite_file} {nsys_file} --force-overwrite true"

    if not run_command(export_cmd, verbose=False):
        print(f"  ERROR: Failed to export {nsys_file} to sqlite")
        return None

    # Then generate txt report from sqlite
    txt_file = f"{OUTPUT_DIR}/nsys_{kernel_key}.txt"
    cmd = f"nsys stats {sqlite_file} > {txt_file} 2>&1"

    if run_command(cmd, verbose=False):
        print(f"  ✓ Exported to TXT: {txt_file}")
        return txt_file
    return None

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("Attention Kernel Profiling Suite")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {OUTPUT_DIR}")

    results = {
        'flash_tc': {'binary': None, 'nsys': None, 'nsys_txt': None, 'ncu': None},
        'flash_no_tc': {'binary': None, 'nsys': None, 'nsys_txt': None, 'ncu': None},
        'classic': {'binary': None, 'nsys': None, 'nsys_txt': None, 'ncu': None}
    }

    # Build all benchmarks
    print("\n" + "="*70)
    print("PHASE 1: Building Benchmarks")
    print("="*70)

    for key, info in KERNELS.items():
        binary = build_benchmark(key, info)
        if binary:
            results[key]['binary'] = binary
        else:
            print(f"WARNING: Failed to build {key}, skipping profiling")

    # Profile all benchmarks
    print("\n" + "="*70)
    print("PHASE 2: Running nsys Profiling")
    print("="*70)

    for key, info in KERNELS.items():
        if results[key]['binary']:
            nsys_file = run_nsys_profile(results[key]['binary'], key, info)
            results[key]['nsys'] = nsys_file

            # Export nsys results to TXT
            if nsys_file:
                nsys_txt = export_nsys_results(nsys_file, key)
                results[key]['nsys_txt'] = nsys_txt

    print("\n" + "="*70)
    print("PHASE 3: Running ncu Profiling")
    print("="*70)

    for key, info in KERNELS.items():
        if results[key]['binary']:
            ncu_file = run_ncu_profile(results[key]['binary'], key, info)
            results[key]['ncu'] = ncu_file

            # Export ncu results to CSV
            if ncu_file:
                export_ncu_results(ncu_file, key)


if __name__ == '__main__':
    main()
