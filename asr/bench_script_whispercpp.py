#!/usr/bin/env python3

import os
import subprocess
import sys
import re
import argparse

def run_command(command: str, cwd=None) -> str:
    try:
        print(f"Running command: {command}")
        result = subprocess.run(command, cwd=cwd, check=True, text=True, shell=True, capture_output=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.strip()}")
        sys.exit(1)

def get_num_cores() -> int:
    print("Getting CPU information...")
    lscpu_output = run_command("lscpu")
    for line in lscpu_output.splitlines():
        if line.startswith("CPU(s):"):
            num_cores = int(line.split(":")[1].strip())
            print(f"Number of cores: {num_cores}")
            return num_cores
    raise RuntimeError("Failed to retrieve the number of CPU cores")

def parse_memcpy_output(output: str) -> dict: 
    """Parse the memcpy bandwidth test output."""
    results = {}
    for line in output.splitlines():
        match = re.match(r"memcpy:\s+([\d\.]+) GB/s \(([\d\w\- ]+)\)", line)
        if match:
            bandwidth = float(match.group(1))
            description = match.group(2).strip()
            results[description] = bandwidth

    print(f"Memcpy results: {results}")
    return results

def parse_matmul_output(output: str) -> dict:
    """Parse the ggml matrix multiplication test output."""
    results = {}
    for line in output.splitlines():
        match = re.findall(r"(\d+ x\s+\d+):\s+([A-Z0-9_]+)\s+([\d\.]+) GFLOPS", line)
        if match:
            for size, format_type, gflops in match:
                if size not in results:
                    results[size] = {}
                results[size][format_type] = float(gflops)

    print(f"Matrix multiplication results: {results}")
    return results

def parse_full_benchmark_output(output: str) -> dict:
    """Parse the full benchmark output."""
    timings = {}
    system_info = {}

    for line in output.splitlines():
        if "system_info" in line:
            key_values = line.split(":")[1].split("|")
            for key_value in key_values:
                key, value = key_value.split("=")
                system_info[key.strip()] = value.strip()
        elif "whisper_print_timings" in line:
            key_value = line.split(":")
            key = key_value[0].strip()
            value = float(key_value[1].strip().split(" ")[0])
            timings[key] = value

    print(f"System info: {system_info}")
    print(f"Timings: {timings}")

    return {"timings": timings, "system_info": system_info}

def run_benchmark(command: str, test_id: int) -> str:
    if test_id == 1:
        print("Running memcpy bandwidth test...")
    elif test_id == 2:
        print("Running ggml matrix multiplication test...")
    else:
        print("Running full benchmark...")

    return run_command(command, cwd=os.path.expanduser("~/repos/whisper.cpp"))

def generate_markdown_table(machine_name: str, memcpy_data: dict, matrix_data: dict, full_benchmark_data: dict) -> str:
    # Create the table header
    table_header = (
        "| Machine   | Memcpy Heat-up (GB/s) | Memcpy 1 Thread (GB/s) | Memcpy 2 Threads (GB/s) | Matrix Size | Q4_0 (GFLOPS) | Q4_1 (GFLOPS) | Q5_0 (GFLOPS) | Q5_1 (GFLOPS) | Q8_0 (GFLOPS) | F16 (GFLOPS) | F32 (GFLOPS) | Encode Time (ms) | Decode Time (ms) | Batch Time (ms) | Prompt Time (ms) | Total Time (ms) |\n"
        "|-----------|-----------------------|------------------------|-------------------------|-------------|---------------|---------------|--------------|--------------|--------------|--------------|--------------|------------------|------------------|-----------------|-----------------|-----------------|\n"
    )

    # Extract Memcpy data
    memcpy_heatup = memcpy_data.get('heat-up', 'N/A')
    memcpy_1_thread = memcpy_data.get('1 thread', 'N/A')
    memcpy_2_threads = memcpy_data.get('2 thread', 'N/A')

    # Iterate over matrix data and full benchmark data to fill the table rows
    table_rows = ""
    for size, gflops_data in matrix_data.items():
        row = (
            f"| {machine_name} "
            f"| {memcpy_heatup} "
            f"| {memcpy_1_thread} "
            f"| {memcpy_2_threads} "
            f"| {size} "
            f"| {gflops_data.get('Q4_0', 'N/A')} "
            f"| {gflops_data.get('Q4_1', 'N/A')} "
            f"| {gflops_data.get('Q5_0', 'N/A')} "
            f"| {gflops_data.get('Q5_1', 'N/A')} "
            f"| {gflops_data.get('Q8_0', 'N/A')} "
            f"| {gflops_data.get('F16', 'N/A')} "
            f"| {gflops_data.get('F32', 'N/A')} "
            f"| {full_benchmark_data['encode_time']} "
            f"| {full_benchmark_data['decode_time']} "
            f"| {full_benchmark_data['batch_time']} "
            f"| {full_benchmark_data['prompt_time']} "
            f"| {full_benchmark_data['total_time']} |\n"
        )
        # Clear the machine name after the first row for readability
        machine_name = ""

        table_rows += row

    return table_header + table_rows

def main():
    parser = argparse.ArgumentParser(description="Run the whisper.cpp benchmarks.")
    # add named arguments for the script
    parser.add_argument("--models_base_dir", help="Base directory containing the models.")
    parser.add_argument("--repos_base_dir", help="Base directory containing the repositories.")
    
    args = parser.parse_args()


    # Step 1: Change to the whisper.cpp directory
    os.chdir(os.path.expanduser("~/repos/whisper.cpp"))

    # Step 2: Get the number of CPU cores
    num_cores = get_num_cores()

    # Step 3: Run the benchmarks and capture the data
    benchmark_results = {}

    # get model file path
    model_file = "~/models/ggml-large-v3.bin"

    try:
        # Benchmark 1: Memcpy bandwidth test
        command = f"./bench -m {model_file} -t {num_cores} -ng -w 1"
        memcpy_output = run_benchmark(command, 1)
        benchmark_results['memcpy_bandwidth_test'] = parse_memcpy_output(memcpy_output)

        # Benchmark 2: GGML matrix multiplication test (ggml_mul_mat)
        command = f"./bench -m {model_file} -t {num_cores} -ng -w 2"
        matmul_output = run_benchmark(command, 2)
        benchmark_results['ggml_matrix_multiplication_test'] = parse_matmul_output(matmul_output)

        # Benchmark 3: Full benchmark with <num_cores> cores
        command = f"./bench -m {model_file} -t {num_cores} -ng"
        full_benchmark_output = run_benchmark(command, 0)
        benchmark_results['full_benchmark'] = parse_full_benchmark_output(full_benchmark_output)

    except Exception as e:
        print(f"An error occurred during benchmarking: {e}")
        sys.exit(1)

    # Step 7: Aggregate the data for all 3 tests
    print("\nBenchmarking completed successfully. Aggregating results...")
    results = {
        "num_cores": num_cores,
        "benchmarks": benchmark_results
    }

    # Generate the markdown table for a single machine with the machine name set as "N/A"
    markdown_table = generate_markdown_table("N/A", 
                                             results['benchmarks']['memcpy_bandwidth_test'], 
                                             results['benchmarks']['ggml_matrix_multiplication_test'], 
                                             {
                                                "encode_time": results['benchmarks']['full_benchmark']['timings'].get('whisper_print_timings:   encode time', 'N/A'),
                                                "decode_time": results['benchmarks']['full_benchmark']['timings'].get('whisper_print_timings:   decode time', 'N/A'),
                                                "batch_time": results['benchmarks']['full_benchmark']['timings'].get('whisper_print_timings:   batchd time', 'N/A'),
                                                "prompt_time": results['benchmarks']['full_benchmark']['timings'].get('whisper_print_timings:   prompt time', 'N/A'),
                                                "total_time": results['benchmarks']['full_benchmark']['timings'].get('whisper_print_timings:    total time', 'N/A')
                                             })

    # Print the aggregated results
    print(markdown_table)

if __name__ == "__main__":
    main()
