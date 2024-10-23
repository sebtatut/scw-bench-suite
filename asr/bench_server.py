import os
import random
import wave
import requests
import time
import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

verbose_level: int = 0

# send and measure request completion time
def send_post_request(url: str, file_path: str, lang: str | None, dur_ms: float) -> tuple[int, float, float]:
    # check params
    if not url:
        raise ValueError('URL is required')
    if not file_path:
        raise ValueError('File path is required')
    if not dur_ms:
        raise ValueError('Clip length is required')
    
    if verbose_level == 1:
        print(f'Requesting transcription for {file_path}, dur {dur_ms}, lang {lang} to {url}')
    
    # open file
    with open(file_path, 'rb') as f:
        # prepare payload
        payload = {'language': lang}

        # start time
        start_time = time.perf_counter()

        try:
            response = requests.post(url, data=payload, files={'file': f})
        except requests.exceptions.RequestException as e:
            print(f'Request failed: {e}')
            return 500, 0, 0
        
        if response.status_code != 200:
            return response.status_code, 0, 0

        # end time
        end_time = time.perf_counter()
    
    # calculate duration in ms
    ptime = (end_time - start_time) * 1000

    # calculate RTF
    rtf = calculate_rtf(ptime, dur_ms)

    if verbose_level == 1:
        print(f"Request completed with status code {response.status_code} in {ptime:.3f} ms with RTF {rtf:.4f}")
    
    return response.status_code, ptime, rtf

# calculate P99 (99th percentile)
def calculate_p99(data: list[float]) -> float:
    # chek params
    if not data:
        raise ValueError('Data is required')

    data_sorted = sorted(data)
    index = math.ceil(0.99 * len(data_sorted)) - 1
    return data_sorted[index]

# calculate median
def calculate_median(data: list[float]) -> float:
    # check params
    if not data:
        raise ValueError('Data is required')

    data_sorted = sorted(data)
    n = len(data_sorted)
    if n % 2 == 0:
        return (data_sorted[n//2 - 1] + data_sorted[n//2]) / 2
    else:
        return data_sorted[n//2]

# calculate mean
def calculate_mean(data: list[float]) -> float:
    # check params
    if not data:
        raise ValueError('Data is required')

    return sum(data) / len(data)

# calculate standard deviation
def calculate_std_dev(data: list[float], mean: float) -> float:
    # check params
    if not data:
        raise ValueError('Data is required')
    if not mean:
        raise ValueError('Mean is required')

    # print(f'data: {data}\tmean: {mean}')
    num_samples = len(data)
    # use num_samples - 1 (Bessel's correction) for sample standard deviation
    if num_samples == 1:
        return 0
    variance = sum((x - mean) ** 2 for x in data) / (num_samples - 1)

    return math.sqrt(variance)

# calculate RTF
def calculate_rtf(iteration_elapsed_time: float, total_clip_time: float) -> float:
    # check params
    if not iteration_elapsed_time:
        raise ValueError('Iteration elapsed time is required')
    if not total_clip_time:
        raise ValueError('Total clip time is required')
    
    return iteration_elapsed_time / total_clip_time

# get clip duration in seconds
def get_clip_dur_ms(file_path: str) -> float:
    # check params
    if not file_path:
        raise ValueError('File path is required')

    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = (frames / float(rate)) * 1000

    return duration

# generate a list of random files to send requests
def generate_random_file_list(file_path: str, requests: int, seed: int) -> list[str]:
    # check params
    if not file_path:
        raise ValueError('File path is required')
    if not requests:
        raise ValueError('Number of requests is required')
    if not seed:
        raise ValueError('Seed is required')
    
    files: list[str] = []
    random.seed(seed)
    
    # check if file_path is file or folder
    if os.path.isdir(file_path):
        # filter out non wav files and files larger than 10MB
        files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.wav') and os.path.getsize(os.path.join(file_path, f)) < 10 * 1024 * 1024]

        if not files:
            raise ValueError('No wav files found in the folder')
        # randomly get requests number of files with allowed duplicates
        files = random.choices(files, k=requests) # type: ignore
    elif os.path.isfile(file_path):
        # repeat 'requests' times
        files = [file_path] * requests
    else:
        raise ValueError('Invalid file path: not a file or directory')
    
    return files

# run benchmark and collect results with batched parallel requests for a single server
def run_benchmark_for_server(url: str, files_to_send: list[str], durations: list[float], lang: str | None, requests: int, request_rate: int) -> list[dict[str, float]]:
    # check params
    if not url:
        raise ValueError('URL is required')
    if not files_to_send or files_to_send == []:
        raise ValueError('Files are required')
    if not durations or durations == []:
        raise ValueError('Durations are required')
    if not requests:
        raise ValueError('Number of requests is required')
    if not request_rate:
        raise ValueError('Request rate is required')
    
    results: list[dict[str, int | float]] = []
        
    # total number of batches
    total_batches = math.ceil(requests / request_rate)
    
    # set up a thread pool for parallel requests
    for batch in range(total_batches):
        with ThreadPoolExecutor(max_workers=request_rate) as executor:
            # determine the number of requests for this batch
            remaining_requests = requests - batch * request_rate
            current_batch_size = min(remaining_requests, request_rate)
            
            # create a list of futures to execute requests
            futures: list[Future[tuple[int, float, float]]] = []
            for i in range(current_batch_size):
                # select file and duration
                file: str = files_to_send[batch * request_rate + i]
                dur: float = durations[batch * request_rate + i]
                futures.append(executor.submit(send_post_request, url, file, lang, dur)) # type: ignore

            # collect the results as each future completes
            for future in as_completed(futures):
                status_code, ptime, rtf = future.result()
                result = {'status': status_code, 'ptime': ptime, 'rtf': rtf} # type: ignore
                if status_code == 200:
                    results.append(result) # type: ignore
                else:
                    print(f'Request failed with status code {status_code}')
    
    return results

# run benchmark for multiple servers concurrently
def run_benchmark(urls: list[str], file_path: str, lang: str | None, requests: int, request_rate: int, iterations: int, seed: int) -> dict[str, list[float] | int]:
    # check params
    if not urls:
        raise ValueError('URLs are required')
    if not file_path:
        raise ValueError('File path is required')
    if not requests:
        raise ValueError('Number of requests is required')
    if not request_rate:
        raise ValueError('Request rate is required')
    if not iterations:
        raise ValueError('Number of iterations is required')

    # init results 
    iterations_ptimes: list[float] = []
    iterations_rtfs: list[float] = []
    total_clip_dur_s: float = 0.0
    total_requests_processed = 0

    # distribute requests across servers
    num_servers = len(urls)
    base_requests_per_server = requests // num_servers
    remainder_requests = requests % num_servers

    # generate a list of files to send requests
    files_to_send: list[str] = generate_random_file_list(file_path, requests, seed)
    # get durations
    durations_ms = [get_clip_dur_ms(file) for file in files_to_send] 
    
    for _ in range(iterations):
        iteration_results: list[dict[str, float]] = []
        
        with ThreadPoolExecutor(max_workers=len(urls)) as server_executor:
            server_futures: list[Future[list[dict[str, float]]]] = []
            for i, url in enumerate(urls):
                # distribute the remainder requests
                server_requests = base_requests_per_server + (1 if i < remainder_requests else 0)
               
                if verbose_level == 1:
                    print(f'Distributed {server_requests} requests to server {url}')
               
                server_futures.append(server_executor.submit(run_benchmark_for_server, url, files_to_send, durations_ms, lang, server_requests, request_rate))

            # collect the results from all servers
            for server_future in as_completed(server_futures):
                server_results = server_future.result()
                iteration_results.extend(server_results)
        
        # update the total number of requests processed
        total_requests_processed += len(iteration_results)
        
        # sum ptimes for this iteration, filter out failed requests
        ptimes = [result['ptime'] for result in iteration_results if result['status'] == 200]
        iterations_ptimes.append(sum(ptimes))

        # accumulate RTFs for this iteration, filter out failed requests
        rtfs = [result['rtf'] for result in iteration_results if result['status'] == 200]
        iterations_rtfs.append(calculate_mean(rtfs))

        # accumulate num failed requests
        failed_requests_num = len(iteration_results) - len(ptimes)
   
        # accumulate transcribed duration, if request was successful
        if ptimes:
            total_clip_dur_s += sum([durations_ms[i] / 1000 for i, result in enumerate(iteration_results) if result['status'] == 200])

    # gather results
    results: dict[str, list[float] | int] = {
        'ptime_sums': iterations_ptimes,
        'rtfs': iterations_rtfs,
        'total_requests': total_requests_processed,
        'failed_requests': failed_requests_num, # type: ignore
        'total_clip_dur': total_clip_dur_s # type: ignore
    }

    return results

# gen statistics and print results
def print_statistics(data: dict[str, list[float] | int], num_servers: int, iterations: int) -> None:
    # check params
    if not data:
        raise ValueError('Data is required')
    if not num_servers:
        raise ValueError('Number of servers is required')
    if not iterations:
        raise ValueError('Number of iterations is required')

    # mean processing time
    mean_time_across_iterations = calculate_mean(data['ptime_sums']) # type: ignore
    std_dev_mean_time = calculate_std_dev(data['ptime_sums'], mean_time_across_iterations) # type: ignore
    # std dev %
    std_dev_mean_time_percentage = (std_dev_mean_time / mean_time_across_iterations) * 100

    # median processing time
    median_time_across_iterations = calculate_median(data['ptime_sums']) # type: ignore

    # P99 processing time
    p99_across_iterations = calculate_p99(data['ptime_sums']) # type: ignore

    # mean RTF
    mean_rtf = calculate_mean(data['rtfs']) # type: ignore
    std_dev_rtf = calculate_std_dev(data['rtfs'], mean_rtf) # type: ignore
    # std dev %
    std_dev_rtf_percentage = (std_dev_rtf / mean_rtf) * 100

    total_elapsed_time_s = sum(data['ptime_sums']) / 1000  # type: ignore # convert to seconds

    # total clip duration across all iterations
    total_clip_dur_s: float = data['total_clip_dur']  # type: ignore
    # converted to hh:mm:ss.ms
    total_clip_dur_hms = time.strftime('%H:%M:%S', time.gmtime(total_clip_dur_s))

    # print the results in table format
    print(f"{'Number of Servers':<40}: {num_servers}")
    print(f"{'Number of Iterations':<40}: {iterations}")
    print(f"{'Total Requests Processed':<40}: {data['total_requests']}")
    print(f"{'Failed Requests':<40}: {data['failed_requests']}")
    print(f"{'Iter. Mean Processing Time (ms)':<40}: {mean_time_across_iterations:.3f}")
    print(f"{'Iter. Median Processing Time (ms)':<40}: {median_time_across_iterations:.3f}")
    print(f"{'Iter. P99 Processing Time (ms)':<40}: {p99_across_iterations:.3f}")
    print(f"{'Iter. Std Dev Processing Time (%)':<40}: {std_dev_mean_time_percentage:.6f}")
    print(f"{'Mean RTF':<40}: {mean_rtf:.6f}")
    print(f"{'Std Dev RTF (%)':<40}: {std_dev_rtf_percentage:.6f}")
    print(f"{'Total Elapsed Time (s)':<40}: {total_elapsed_time_s:.3f}")
    print(f"{'Total Clip Durations (hh:mm:ss.ms)':<40}: {total_clip_dur_hms}")

# setup arg parser
def main():
    parser = argparse.ArgumentParser(description='ASR server benchmark')
    parser.add_argument('--urls', nargs='+', type=str, required=True, help='List of endpoint URLs to send the POST request to')
    parser.add_argument('--files', type=str, required=True, help='Path to the file to be uploaded')
    parser.add_argument('--lang', type=str, required=False, help='Audio file language')
    parser.add_argument('--total-requests', type=int, required=True, help='Total number of requests to be distributed across servers')
    parser.add_argument('--request-rate', type=int, required=True, help='Requests per second per server')
    parser.add_argument('--runs', type=int, required=True, help='Number of benchmark iterations (runs)')
    parser.add_argument('--seed', type=int, required=False, default=0, help='Seed for random file selection')
    parser.add_argument('--verbose', type=int, required=False, default=0, help='Verbose level (0, 1)')

    args = parser.parse_args()

    # set verbose level
    global verbose_level
    verbose_level = args.verbose

    # run bench
    data = run_benchmark(args.urls, args.files, args.lang, args.total_requests, args.request_rate, args.runs, args.seed)

    # print statistics
    print_statistics(data, len(args.urls), args.runs)

if __name__ == '__main__':
    main()
