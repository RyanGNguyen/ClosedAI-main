#!/usr/bin/env python3
"""
Loop Unrolling Optimization Configuration Sweep Script
This script systematically tests different loop unrolling factors and other
configurations to find the optimal performance for the Flash Attention kernel.

Strategy:
- Test different unrolling factors for key computation loops (1, 2, 4, 8)
- Sweep block sizes and thread counts
- Profile each configuration using nsys
- Analyze results to identify the best configuration and understand trade-offs
"""

import subprocess
import os
import sys
import json
import csv
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import shutil

# Configuration for sweep
NVCC = 'nvcc'
CFLAGS = '-O3 -arch=sm_86 -std=c++11 -rdc=true -g -lineinfo'
CUBLAS_INCLUDES = '-I/usr/local/cuda/include'
CUBLAS_LIBS = '-L/usr/local/cuda/lib64 -lcublas -lcublasLt -lnvToolsExt'

BENCHMARK_SCRIPT = 'kernels_req_5/benchmark_attention.cu'
KERNEL_SOURCE = 'kernels/attention.cuh'
WORK_DIR = 'kernels_req_5'
TEMP_DIR = f'{WORK_DIR}/temp_kernels'
RESULTS_DIR = f'{WORK_DIR}/sweep_results'

# Define configuration space to sweep
# These are the primary parameters we'll test
# NOTE: For WMMA operations in the kernel:
#   - D must be divisible by 8 (WMMA_K)
#   - KV_TILE_SIZE must be divisible by 16 (WMMA_N) for compute_qk
#   - Q_TILE_SIZE must be divisible by 16 (WMMA_M) for both compute_qk and compute_pv
# Only testing WMMA-compatible sizes to avoid performance anomalies
UNROLL_FACTORS = [16]  # Loop unrolling factors for computation loops
Q_TILE_SIZE = [8, 16, 32, 64, 128, 256, 512]  # Q_TILE_SIZE variations (must be multiples of 16 for WMMA)
KV_SIZES = [8, 16,32,64,128,256,512]  # KV_TILE_SIZE (must be multiples of 16 for WMMA)
THREAD_COUNTS = [128]  # THREADS per block


class ConfigSweeper:
    """Manages configuration sweep for loop unrolling optimization"""

    def __init__(self):
        self.results = []
        self.temp_dir = Path(TEMP_DIR)
        self.results_dir = Path(RESULTS_DIR)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def setup(self):
        """Create necessary directories"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Setup complete. Using temp dir: {self.temp_dir}")

    def cleanup_temp(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_kernel_variant(self,
                            unroll_qkv: int,
                            unroll_scale: int,
                            unroll_output: int,
                            q_tile: int,
                            kv_tile: int,
                            threads: int) -> Tuple[str, Path]:
        """
        Create a kernel variant with specified unrolling factors and tile sizes.

        Args:
            unroll_qkv: Unroll factor for D-dimension loop in QK computation
            unroll_scale: Unroll factor for scaling/masking loop
            unroll_output: Unroll factor for output computation
            q_tile: Q_TILE_SIZE
            kv_tile: KV_TILE_SIZE
            threads: THREADS per block

        Returns:
            Tuple of (variant_name, kernel_file_path)
        """
        # Read original kernel
        with open(KERNEL_SOURCE, 'r') as f:
            kernel_content = f.read()

        # Create variant name
        variant_name = (f"unroll_{unroll_qkv}_scale_{unroll_scale}_out_{unroll_output}_"
                       f"qtile_{q_tile}_kvtile_{kv_tile}_th_{threads}")

        # Fix include paths for temp directory location
        kernel_content = kernel_content.replace(
            '#include "../utils/cuda_utils.cuh"',
            '#include "../../utils/cuda_utils.cuh"'
        )

        # Modify tile sizes
        kernel_content = kernel_content.replace(
            '#define Q_TILE_SIZE 16',
            f'#define Q_TILE_SIZE {q_tile}'
        )
        kernel_content = kernel_content.replace(
            '#define KV_TILE_SIZE 64',
            f'#define KV_TILE_SIZE {kv_tile}'
        )
        kernel_content = kernel_content.replace(
            '#define THREADS 128',
            f'#define THREADS {threads}'
        )

        # Add unrolling pragmas
        # 1. D-dimension loop in compute_qkv_wmma (line ~79)
        if unroll_qkv > 1:
            kernel_content = kernel_content.replace(
                '            // Loop over D dimension in chunks of WMMA_K (8 for TF32)\n'
                '            for (int d = 0; d < D; d += WMMA_K) {',
                f'            // Loop over D dimension in chunks of WMMA_K (8 for TF32)\n'
                f'            #pragma unroll {unroll_qkv}\n'
                f'            for (int d = 0; d < D; d += WMMA_K) {{'
            )

        # 2. KV dimension loop in compute_pv_wmma (line ~135)
        if unroll_output > 1:
            kernel_content = kernel_content.replace(
                '            // Loop over KV dimension in chunks of WMMA_K (8)\n'
                '            for (int kv = 0; kv < KV_TILE_SIZE; kv += WMMA_K) {',
                f'            // Loop over KV dimension in chunks of WMMA_K (8)\n'
                f'            #pragma unroll {unroll_output}\n'
                f'            for (int kv = 0; kv < KV_TILE_SIZE; kv += WMMA_K) {{'
            )

        # 3. Scaling and masking loop (line ~246)
        if unroll_scale > 1:
            kernel_content = kernel_content.replace(
                '        // Apply softmax scaling and causal masking\n'
                '        for (int qk_idx = tid; qk_idx < KV_TILE_SIZE * Q_TILE_SIZE; qk_idx += blockDim.x) {',
                f'        // Apply softmax scaling and causal masking\n'
                f'        #pragma unroll {unroll_scale}\n'
                f'        for (int qk_idx = tid; qk_idx < KV_TILE_SIZE * Q_TILE_SIZE; qk_idx += blockDim.x) {{'
            )

        # 4. Existing unroll in compute_pv_wmma fallback
        if unroll_output > 1:
            kernel_content = kernel_content.replace(
                '                    #pragma unroll 4\n'
                '                    for (int kv_idx = 0; kv_idx < KV_TILE_SIZE; kv_idx++) {',
                f'                    #pragma unroll {unroll_output}\n'
                f'                    for (int kv_idx = 0; kv_idx < KV_TILE_SIZE; kv_idx++) {{'
            )

        # Write to temp location
        kernel_file = self.temp_dir / f'attention_{variant_name}.cuh'
        with open(kernel_file, 'w') as f:
            f.write(kernel_content)

        return variant_name, kernel_file

    def compile_benchmark(self,
                         kernel_file: Path,
                         variant_name: str) -> Optional[str]:
        """
        Compile benchmark with specified kernel variant.

        Returns:
            Binary name if successful, None otherwise
        """
        binary_name_short = f'benchmark_{variant_name}'
        object_file_short = f'benchmark_{variant_name}.o'
        binary_path = self.temp_dir / binary_name_short
        object_file_path = self.temp_dir / object_file_short

        # Create a temporary benchmark file that includes our kernel
        with open(BENCHMARK_SCRIPT, 'r') as f:
            bench_content = f.read()

        # Replace kernel include path with relative path
        relative_kernel_path = kernel_file.name
        bench_content = bench_content.replace(
            '#include "kernels/attention.cuh"',
            f'#include "{relative_kernel_path}"'
        )

        bench_file = self.temp_dir / f'benchmark_{variant_name}.cu'
        with open(bench_file, 'w') as f:
            f.write(bench_content)

        # Compile (from temp directory to keep paths local)
        compile_cmd = (
            f"cd {self.temp_dir} && "
            f"{NVCC} {CFLAGS} {CUBLAS_INCLUDES} "
            f"-I../../ -c {bench_file.name} -o {object_file_short}"
        )

        try:
            result = subprocess.run(compile_cmd, shell=True,
                                  capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"  X Compilation failed for {variant_name}")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")
                return None
        except subprocess.TimeoutExpired:
            print(f"  X Compilation timeout for {variant_name}")
            return None

        # Link (from temp directory)
        link_cmd = (
            f"cd {self.temp_dir} && "
            f"{NVCC} {CFLAGS} -o {binary_name_short} {object_file_short} {CUBLAS_LIBS}"
        )
        try:
            result = subprocess.run(link_cmd, shell=True,
                                  capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"  X Linking failed for {variant_name}")
                return None
        except subprocess.TimeoutExpired:
            print(f"  X Linking timeout for {variant_name}")
            return None

        return str(binary_path)

    def benchmark_kernel(self,
                        binary_path: str,
                        variant_name: str) -> Optional[Dict]:
        """
        Run benchmark for a kernel variant and capture timing.

        Returns:
            Dict with timing information or None if failed
        """
        try:
            # Get just the binary name for execution
            binary_name = binary_path.split('/')[-1]
            result = subprocess.run(
                f"cd {self.temp_dir} && ./{binary_name}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                print(f"  X Execution failed for {variant_name}")
                return None

            # Parse output to extract timing information
            output = result.stdout
            timing_info = self._parse_benchmark_output(output)

            if timing_info is None:
                print(f"  X Could not parse timing for {variant_name}")
                return None

            return timing_info

        except subprocess.TimeoutExpired:
            print(f"  X Benchmark timeout for {variant_name}")
            return None

    def _parse_benchmark_output(self, output: str) -> Optional[Dict]:
        """Parse benchmark output to extract timing metrics"""
        import re
        import math

        timing_dict = {}

        # Look for timing line: "Time (ms): avg=X.XXXX, min=X.XXXX, max=X.XXXX"
        # Allow for scientific notation and special values (inf, nan)
        time_match = re.search(r'Time \(ms\): avg=([\d.eE+-]+), min=([\d.eE+-]+), max=([\d.eE+-]+)', output)
        if time_match:
            try:
                avg_time = float(time_match.group(1))
                min_time = float(time_match.group(2))
                max_time = float(time_match.group(3))

                # Validate timing values
                if math.isnan(avg_time) or math.isinf(avg_time) or avg_time <= 0:
                    print(f"    WARNING: Invalid timing value: avg={avg_time}")
                    return None
                if avg_time > 10000:  # More than 10 seconds is suspicious
                    print(f"    WARNING: Timing unusually high: {avg_time}ms")
                    return None

                timing_dict['avg_time'] = avg_time
                timing_dict['min_time'] = min_time
                timing_dict['max_time'] = max_time
            except (ValueError, OverflowError):
                print(f"    WARNING: Could not parse timing values")
                return None
        else:
            return None

        # Look for throughput: "Throughput: X.XX GFLOPS"
        throughput_match = re.search(r'Throughput: ([\d.eE+-]+) GFLOPS', output)
        if throughput_match:
            try:
                throughput = float(throughput_match.group(1))
                # Validate throughput (must be positive and not NaN/Inf)
                if not math.isnan(throughput) and not math.isinf(throughput) and throughput > 0:
                    timing_dict['throughput'] = throughput
                else:
                    print(f"    WARNING: Invalid throughput: {throughput} GFLOPS")
                    timing_dict['throughput'] = 0.0
            except (ValueError, OverflowError):
                timing_dict['throughput'] = 0.0

        return timing_dict if timing_dict else None

    def run_sweep(self):
        """Execute the configuration sweep"""
        print("\n" + "="*80)
        print("LOOP UNROLLING CONFIGURATION SWEEP")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sweeping unroll factors: {UNROLL_FACTORS}")
        print(f"Sweeping block sizes: {Q_TILE_SIZE}")
        print(f"Sweeping KV sizes: {KV_SIZES}")
        print(f"Sweeping thread counts: {THREAD_COUNTS}")

        total_configs = (len(UNROLL_FACTORS) *
                        len(Q_TILE_SIZE) * len(KV_SIZES) * len(THREAD_COUNTS))
        print(f"Total configurations to test: {total_configs}")
        print("="*80)

        config_num = 0

        # Iterate through all configurations
        for unroll_factor in UNROLL_FACTORS:
            for q_tile in Q_TILE_SIZE:
                for kv_tile in KV_SIZES:
                    for threads in THREAD_COUNTS:
                        config_num += 1

                        # Skip some configurations to reduce test time
                        # (optional - remove for comprehensive sweep)
                        # if config_num % 2 != 0:
                        #     continue

                        config_dict = {
                            'config_num': config_num,
                            'unroll_qkv': unroll_factor,
                            'unroll_scale': unroll_factor,
                            'unroll_output': unroll_factor,
                            'q_tile_size': q_tile,
                            'kv_tile_size': kv_tile,
                            'attn_threads': threads,
                        }

                        self._test_configuration(config_dict, config_num, total_configs)

        # Save results
        self._save_results()

    def _test_configuration(self, config: Dict, config_num: int, total: int):
        """Test a single configuration"""
        print(f"\n[{config_num}/{total}] Testing: "
              f"unroll({config['unroll_qkv']},{config['unroll_scale']},{config['unroll_output']}) "
              f"qtile={config['q_tile_size']} kvtile={config['kv_tile_size']} "
              f"threads={config['attn_threads']}")

        # Step 1: Create kernel variant
        variant_name, kernel_file = self.create_kernel_variant(
            config['unroll_qkv'],
            config['unroll_scale'],
            config['unroll_output'],
            config['q_tile_size'],
            config['kv_tile_size'],
            config['attn_threads']
        )

        # Step 2: Compile
        print(f"  Compiling...", end=' ', flush=True)
        binary = self.compile_benchmark(kernel_file, variant_name)

        if binary is None:
            print("X Failed")
            self.results.append({
                **config,
                'success': False,
                'compile_time': -1,
                'execution_time': -1,
                'throughput': -1
            })
            return

        print("Done")

        # Step 3: Benchmark
        print(f"  Benchmarking...", end=' ', flush=True)
        timing = self.benchmark_kernel(binary, variant_name)

        if timing is None:
            print("X Failed")
            self.results.append({
                **config,
                'success': False,
                'compile_time': 0,
                'execution_time': -1,
                'throughput': -1
            })
            return

        print("Done")

        # Store results
        result_dict = {
            **config,
            'success': True,
            'compile_time': 0,  # Not tracking compile time in detail
            'execution_time': timing['avg_time'],
            'throughput': timing.get('throughput', 0)
        }

        self.results.append(result_dict)

        # Print summary
        print(f"    Time: {timing['avg_time']:.4f}ms, "
              f"Throughput: {timing.get('throughput', 0):.2f} GFLOPS")

    def _save_results(self):
        """Save results to CSV and JSON"""
        if not self.results:
            print("No results to save!")
            return

        # Sort results by execution time
        sorted_results = sorted(
            [r for r in self.results if r['success']],
            key=lambda x: x['execution_time']
        )

        # Save to CSV
        csv_file = self.results_dir / f'sweep_results_{self.timestamp}.csv'
        with open(csv_file, 'w', newline='') as f:
            fieldnames = [
                'config_num', 'unroll_qkv', 'unroll_scale', 'unroll_output',
                'q_tile_size', 'kv_tile_size', 'attn_threads',
                'success', 'compile_time', 'execution_time', 'throughput'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"\nResults saved to: {csv_file}")

        # Save to JSON
        json_file = self.results_dir / f'sweep_results_{self.timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to: {json_file}")

        # Print summary
        self._print_summary(sorted_results)

    def _print_summary(self, sorted_results: List[Dict]):
        """Print summary of sweep results"""
        print("\n" + "="*80)
        print("SWEEP SUMMARY")
        print("="*80)

        total = len(self.results)
        successful = len([r for r in self.results if r['success']])
        failed = total - successful

        print(f"\nTotal configurations tested: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {100*successful/total:.1f}%")

        if not sorted_results:
            print("\nNo successful configurations!")
            return

        # Find baseline (all default values: unroll=1, qtile=16, kvtile=64, threads=128)
        baseline = None
        for r in sorted_results:
            if (r['unroll_qkv'] == 1 and r['unroll_scale'] == 1 and r['unroll_output'] == 1 and
                r['q_tile_size'] == 16 and r['kv_tile_size'] == 64 and r['attn_threads'] == 128):
                baseline = r
                break

        print(f"\n{'-'*80}")
        print("BEST CONFIGURATION (Fastest):")
        print(f"{'-'*80}")
        best = sorted_results[0]
        print(f"Execution Time: {best['execution_time']:.4f} ms")
        if baseline:
            speedup = baseline['execution_time'] / best['execution_time']
            print(f"Speedup vs baseline: {speedup:.2f}x")
        print(f"\nParameters:")
        print(f"  Unroll QKV: {best['unroll_qkv']}")
        print(f"  Unroll Scale: {best['unroll_scale']}")
        print(f"  Unroll Output: {best['unroll_output']}")
        print(f"  Q_TILE_SIZE: {best['q_tile_size']}")
        print(f"  KV_TILE_SIZE: {best['kv_tile_size']}")
        print(f"  THREADS: {best['attn_threads']}")

        if baseline:
            print(f"\n{'-'*80}")
            print("BASELINE CONFIGURATION (Default):")
            print(f"{'-'*80}")
            print(f"Execution Time: {baseline['execution_time']:.4f} ms")

        # Top 10 configurations
        print(f"\n{'-'*80}")
        print("TOP 10 CONFIGURATIONS:")
        print(f"{'-'*80}")
        for i, result in enumerate(sorted_results[:10], 1):
            speedup_str = ""
            if baseline:
                speedup = baseline['execution_time'] / result['execution_time']
                speedup_str = f" ({speedup:.2f}x)"
            print(f"{i:2d}. {result['execution_time']:8.4f}ms{speedup_str} - "
                  f"unroll({result['unroll_qkv']},{result['unroll_scale']},{result['unroll_output']}) "
                  f"tile({result['q_tile_size']},{result['kv_tile_size']}) "
                  f"threads={result['attn_threads']}")


def main():
    """Main entry point"""
    sweeper = ConfigSweeper()

    try:
        sweeper.setup()
        sweeper.run_sweep()
    finally:
        sweeper.cleanup_temp()


if __name__ == '__main__':
    main()
