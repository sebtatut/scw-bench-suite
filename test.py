import requests
import concurrent.futures
import argparse
import time

def send_post_request(ip, port, file_path):
    url = f"http://{ip}:{port}/inference"
    files = {'file': open(file_path, 'rb')}
    
    try:
        start_time = time.time()
        response = requests.post(url, files=files)
        end_time = time.time()
        response_time = end_time - start_time
        print(f"Response from {url}: {response.status_code} - {response.text[:50]}... | Response Time: {response_time:.2f} seconds")
        return response_time
    except Exception as e:
        print(f"Failed to connect to {url}: {e}")
        return None

def run_requests(ip, file_path, ports, num_runs):
    all_times = []
    
    for run in range(num_runs):
        print(f"Starting run {run + 1} of {num_runs}")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(send_post_request, ip, port, file_path) for port in ports]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            run_times = [r for r in results if r is not None]
            all_times.extend(run_times)
        
        print(f"Completed run {run + 1} of {num_runs}")
        print(f"Run {run + 1} statistics:")
        print(f"  - Total requests: {len(run_times)}")
        print(f"  - Average response time: {sum(run_times)/len(run_times):.2f} seconds" if run_times else "No successful requests")
    
    if all_times:
        print("\nOverall statistics:")
        print(f"  - Total requests sent: {len(all_times)}")
        print(f"  - Average response time: {sum(all_times)/len(all_times):.2f} seconds")
        print(f"  - Minimum response time: {min(all_times):.2f} seconds")
        print(f"  - Maximum response time: {max(all_times):.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send POST requests to multiple servers in parallel.")
    parser.add_argument("ip", help="The IP address of the servers.")
    parser.add_argument("file_path", help="The path to the file to be sent.")
    parser.add_argument("--ports", nargs='+', type=int, default=[8080, 8181, 8282, 8383],
                        help="List of ports to send requests to. Default is 8080, 8181, 8282, 8383.")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to repeat the requests. Default is 1.")
    
    args = parser.parse_args()

    run_requests(args.ip, args.file_path, args.ports, args.runs)
