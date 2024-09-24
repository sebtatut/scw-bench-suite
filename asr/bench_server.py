import requests
import time
import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# function to measure request time and send file
def send_post_request(url: str, file_path: str, lang: str | None) -> tuple[int, float]:
    # open the file in binary mode
    # prepare payload
    with open(file_path, 'rb') as f:
        # prepare payload
        payload = {'lang': lang}

        # start time
        start_time = time.time()
        # send POST request with file as payload
        response = requests.post(url, data=payload, files={'file': f})
        # end time
        end_time = time.time()
    
    # calculate duration in ms
    duration_ms = (end_time - start_time) * 1000
    return response.status_code, duration_ms

# calculate P99 (99th percentile)
def calculate_p99(data: list[float]) -> float:
    data_sorted = sorted(data)
    index = math.ceil(0.99 * len(data_sorted)) - 1
    return data_sorted[index]

# calculate median
def calculate_median(data: list[float]) -> float:
    data_sorted = sorted(data)
    n = len(data_sorted)
    if n % 2 == 0:
        return (data_sorted[n//2 - 1] + data_sorted[n//2]) / 2
    else:
        return data_sorted[n//2]

# calculate mean
def calculate_mean(data: list[float]) -> float:
    return sum(data) / len(data)

# calculate standard deviation
def calculate_std_dev(data: list[float], mean: float) -> float:
    print(f"data: {data}\tmean: {mean}")
    # use len(data) - 1 for sample standard deviation
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

# calculate RTF
def calculate_rtf(iteration_elapsed_time: float, total_clip_time: float) -> float:
    return iteration_elapsed_time / total_clip_time

# run benchmark and collect results with batched parallel requests for a single server
def run_benchmark_for_server(url: str, file_path: str, lang: str | None, requests: int, request_rate: int) -> list[float]:
    results = []
    
    # total number of batches
    total_batches = math.ceil(requests / request_rate)
    
    # set up a thread pool for parallel requests
    for batch in range(total_batches):
        with ThreadPoolExecutor(max_workers=request_rate) as executor:
            # determine the number of requests for this batch
            remaining_requests = requests - batch * request_rate
            current_batch_size = min(remaining_requests, request_rate)
            
            # create a list of futures to execute requests
            futures = []
            for _ in range(current_batch_size):
                futures.append(executor.submit(send_post_request, url, file_path, lang))

            # collect the results as each future completes
            for future in as_completed(futures):
                status_code, duration = future.result()
                if status_code == 200:
                    results.append(duration)
                else:
                    print(f"Request failed with status code {status_code}")
    
    return results

# run benchmark for multiple servers concurrently
def run_benchmark(urls: list[str], file_path: str, lang: str | None, clip_length: float, requests: int, request_rate: int, iterations: int) -> tuple[list[float], list[float], list[float], int]:
    mean_processing_times_per_iteration = []
    median_per_iteration = []
    p99_per_iteration = []
    rtf_per_iteration = []
    total_requests_processed = 0

    per_server_requests = math.ceil(requests / len(urls))  # distribute requests across servers
    
    for _ in range(iterations):
        iteration_results = []
        
        # record iteration start time
        iteration_start_time = time.time()

        with ThreadPoolExecutor(max_workers=len(urls)) as server_executor:
            # submit tasks for each server
            server_futures = [server_executor.submit(run_benchmark_for_server, url, file_path, lang, per_server_requests, request_rate) for url in urls]

            # collect the results from all servers
            for server_future in as_completed(server_futures):
                server_results = server_future.result()
                iteration_results.extend(server_results)
        
        # update the total number of requests processed
        total_requests_processed += len(iteration_results)

        # record iteration end time
        iteration_end_time = time.time()
        
        # calculate mean, median, and P99 for this iteration
        mean_processing_time = calculate_mean(iteration_results)
        median_processing_time = calculate_median(iteration_results)
        p99_processing_time = calculate_p99(iteration_results)
        
        mean_processing_times_per_iteration.append(mean_processing_time)
        median_per_iteration.append(median_processing_time)
        p99_per_iteration.append(p99_processing_time)

        # calculate RTF for this iteration
        iteration_elapsed_time = (iteration_end_time - iteration_start_time) * 1000  # in ms
        total_clip_time = requests * clip_length * 1000  # in ms
        rtf = calculate_rtf(iteration_elapsed_time, total_clip_time)
        rtf_per_iteration.append(rtf)

    return mean_processing_times_per_iteration, median_per_iteration, p99_per_iteration, rtf_per_iteration, total_requests_processed

# generate statistics and print results in a simple table
def print_statistics(mean_times: list[float], median_times: list[float], p99_times: list[float], rtfs: list[float], total_requests: int, total_elapsed_time: float, num_servers: int, iterations: int) -> None:
    # mean processing time and its standard deviation
    mean_time_across_iterations = calculate_mean(mean_times)
    std_dev_mean_time = calculate_std_dev(mean_times, mean_time_across_iterations)
    # std dev %
    std_dev_mean_time_percentage = (std_dev_mean_time / mean_time_across_iterations) * 100

    # median processing time
    median_time_across_iterations = calculate_median(median_times)

    # P99 processing time
    p99_across_iterations = calculate_p99(p99_times)

    # mean RTF and its standard deviation
    mean_rtf = calculate_mean(rtfs)
    std_dev_rtf = calculate_std_dev(rtfs, mean_rtf)
    # std dev %
    std_dev_rtf_percentage = (std_dev_rtf / mean_rtf) * 100

    total_elapsed_time_s = total_elapsed_time / 1000  # convert to seconds

    # print the results in table format
    print(f"{'Number of Servers':<30}: {num_servers}")
    print(f"{'Number of Iterations':<30}: {iterations}")
    print(f"{'Total Requests Processed':<30}: {total_requests}")
    print(f"{'Mean Processing Time (ms)':<30}: {mean_time_across_iterations:.2f}")
    print(f"{'Median Processing Time (ms)':<30}: {median_time_across_iterations:.2f}")
    print(f"{'P99 Processing Time (ms)':<30}: {p99_across_iterations:.2f}")
    print(f"{'Std Dev Processing Time (%)':<30}: {std_dev_mean_time_percentage:.4f}")
    print(f"{'Mean RTF':<30}: {mean_rtf:.2f}")
    print(f"{'Std Dev RTF (%)':<30}: {std_dev_rtf_percentage:.4f}")
    print(f"{'Total Elapsed Time (s)':<30}: {total_elapsed_time_s:.2f}")

# set up argument parser
def main():
    parser = argparse.ArgumentParser(description='Benchmark script for POST requests with file upload and transcription')
    parser.add_argument('--urls', nargs='+', type=str, required=True, help='List of endpoint URLs to send the POST request to')
    parser.add_argument('--file', type=str, required=True, help='Path to the file to be uploaded')
    parser.add_argument('--lang', type=str, required=False, help='Audio file language')
    parser.add_argument('--total-requests', type=int, required=True, help='Total number of requests to be distributed across servers')
    parser.add_argument('--clip-length', type=float, required=True, help='Clip length in seconds')
    parser.add_argument('--request-rate', type=int, required=True, help='Requests per second per server')
    parser.add_argument('--runs', type=int, required=True, help='Number of benchmark iterations (runs)')

    args = parser.parse_args()

    # start time for the entire benchmark
    benchmark_start_time = time.time()

    # run the benchmark across multiple servers
    mean_times, median_times, p99_times, rtfs, total_requests_processed = run_benchmark(args.urls, args.file, args.lang, args.clip_length, args.total_requests, args.request_rate, args.runs)

    # end time for the entire benchmark
    benchmark_end_time = time.time()

    # calculate total elapsed time in ms
    total_elapsed_time = (benchmark_end_time - benchmark_start_time) * 1000

    if mean_times and rtfs:
        # print statistics, including total elapsed time and number of servers
        print_statistics(mean_times, median_times, p99_times, rtfs, total_requests_processed, total_elapsed_time, len(args.urls), args.runs)
    else:
        print("No successful requests to calculate statistics.")

if __name__ == '__main__':
    main()
