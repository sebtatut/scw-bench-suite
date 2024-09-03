#!/usr/bin/env python3

import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import select

def run_command(command: str, timeout: int = 60) -> tuple:
    try:
        print(f"[DEPLOY] Running command: {command}")
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True, timeout=timeout)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout.strip(), e.stderr.strip(), e.returncode
    except subprocess.TimeoutExpired as e:
        return "", f"Command timed out after {timeout} seconds", 1

def copy_file_to_remote(script_path: str, user_ip: str, remote_path: str) -> bool:
    """Copy a file to a remote machine using scp"""
    if not os.path.exists(script_path):
        print(f"[DEPLOY] Script not found: {script_path}")
        return False
    
    print(f"[DEPLOY] Copying {script_path} to {user_ip} ...")
    
    scp_command = f"scp {script_path} {user_ip}:{remote_path}"
    _, scp_stderr, scp_status = run_command(scp_command)
    
    if scp_status != 0:
        trimmed_error = '\n'.join(scp_stderr.strip().splitlines()[-30:]) if scp_stderr else "N/A"
        print(f"[DEPLOY] Failed to copy script to {user_ip}: {trimmed_error}")
        return False
    
    return True

def make_script_executable(user_ip: str, remote_script_path: str) -> bool:
    """Make a script executable on a remote machine using ssh and chmod"""
    chmod_command = f"ssh {user_ip} 'chmod 755 {remote_script_path}'"
    chmod_stdout, chmod_stderr, chmod_status = run_command(chmod_command)

    if chmod_status != 0:
        trimmed_error = '\n'.join(chmod_stderr.strip().splitlines()[-30:]) if chmod_stderr else "N/A"
        print(f"[DEPLOY] Failed to make the script executable on {user_ip}: {trimmed_error}")
        return False
    
    return True

def execute_install(user_ip: str, machine_type: str, script_paths: list, remote_script_path: str, cuda: bool, cleanup: bool) -> tuple:
    """Execute the install script on a remote machine"""
    # get the fully qualified path of the install script
    remote_script_path = os.path.join(remote_script_path, os.path.basename(script_paths[0]))

    # prepare the command to execute the install script on the remote machine
    install_command = 'install' 
    if cleanup:
        install_command = 'cleanup'

    if cuda: 
        ssh_command = f"""ssh {user_ip} 'python3 -u {remote_script_path} {install_command} --cuda'"""
    else:
        ssh_command = f"""ssh {user_ip} 'python3 -u {remote_script_path} {install_command}'"""
        
    try:
        print(f"[DEPLOY] Running command: {ssh_command}")
        # subprocess to capture the output in real-time
        process = subprocess.Popen(ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        stdout_lines = []
        stderr_lines = []

        # read stdout and stderr in real-time
        while True:
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            ret = select.select(reads, [], [])

            for fd in ret[0]:
                if fd == process.stdout.fileno():
                    output = os.read(fd, 1024).decode('utf-8')
                    print(f"[DEPLOY] Output: {output.strip()}")
                    if "[INSTALL]" in output:
                        print(output.strip())
                        stdout_lines.append(output.strip())
                if fd == process.stderr.fileno():
                    error = os.read(fd, 1024).decode('utf-8')
                    stderr_lines.append(error.strip())

            if process.poll() is not None:
                break

        print(f"[DEPLOY] Process return code: {process.returncode}")

        # capture any remaining stderr output
        stderr_output, _ = process.communicate()
        stderr_lines.extend(stderr_output.strip().splitlines())

        status = process.returncode

        stdout_output = '\n'.join(stdout_lines)
        stderr_output = '\n'.join(stderr_lines)

        if status == 0:
            print(f"[DEPLOY] Successfully executed on {user_ip} ({machine_type})")
            return f"{user_ip}_{machine_type}", {
                "status": "Success",
                "output": stdout_output,
                "error": ""
            }
        else:
            print(f"[DEPLOY] Failed to execute on {user_ip} ({machine_type})")
            # Only return the last 30 lines of the error output to keep it tidy
            trimmed_error = '\n'.join(stderr_output.strip().splitlines()[-30:])
            print(f"[DEPLOY] Error: {trimmed_error}")
            return f"{user_ip}_{machine_type}", {
                "status": "Failed",
                "output": stdout_output,
                "error": trimmed_error
            }

    except Exception as e:
        print(f"[DEPLOY] Exception while executing on {user_ip} ({machine_type}): {e}")
        return f"{user_ip}_{machine_type}", {
            "status": "Failed",
            "output": "",
            "error": str(e)
        }

def process_machine(user_ip: str, machine_type: str, script_paths: list, remote_path: str, cuda: bool, cleanup: bool) -> tuple:
    """Process a single machine: copy the script and execute it"""
    user_ip = user_ip.strip()
    if not user_ip:
        return f"{user_ip}_{machine_type}", {"status": "Skipped", "output": "", "error": "Empty machine entry"}

    print(f"[DEPLOY] Processing {user_ip} ({machine_type})...")

    # copy the script(s) to the remote machine
    for script_path in script_paths:
        # get fully qualified remote script path
        remote_script_path = os.path.join(remote_path, os.path.basename(script_path))

        if not copy_file_to_remote(script_path, user_ip, remote_path):
            return f"{user_ip}_{machine_type}", {
                "status": "Failed",
                "output": "",
                "error": f"[DEPLOY] Failed to copy script to {user_ip} ({machine_type})"
            }
        
        # make the script executable on the remote machine
        if not make_script_executable(user_ip, remote_script_path):
            return f"{user_ip}_{machine_type}", {
                "status": "Failed",
                "output": "",
                "error": f"[DEPLOY] Failed to make the script executable on {user_ip} ({machine_type})"
            }
    
    # execute the install script on the remote machine
    return execute_install(user_ip, machine_type, script_paths, remote_path, cuda, cleanup)

def deploy_and_run_script(script_paths: list, remote_path: str, machines_file: str, cuda: bool, cleanup: bool) -> dict:
    """Deploy and execute the script on multiple remote machines in parallel"""
    with open(machines_file, 'r') as file:
        machines = file.readlines()

    results = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_machine = {}
        for machine in machines:
            user_ip, machine_type = machine.strip().split("_", 1)
            future = executor.submit(process_machine, user_ip, machine_type, script_paths, remote_path, cuda, cleanup)
            future_to_machine[future] = f"{user_ip}_{machine_type}"
        
        for future in as_completed(future_to_machine):
            machine = future_to_machine[future]
            try:
                machine, result = future.result()
                results[machine] = result
            except Exception as exc:
                print(f"[DEPLOY] {machine} generated an exception: {exc}")
                results[machine] = {"status": "Failed", "output": "", "error": str(exc)}

    print("[DEPLOY] Script deployment and execution completed")
    return results

def save_results(results: dict, output_file: str) -> None:
    """Save the gathered results to a file"""
    with open(output_file, 'w') as file:
        for machine, result in results.items():
            file.write(f"Machine: {machine}\n")
            file.write(f"Status: {result['status']}\n")
            if result['status'] == "Success":
                file.write("Output:\n")
                file.write(result['output'])
            else:
                file.write("Error (last 30 lines):\n")
                file.write(result['error'])
            file.write("\n" + "-"*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Deploy and run a script on multiple remote machines")
    # this argument can contain the path of multiple scripts
    parser.add_argument("--scripts", type=str, nargs="+", help="Path(s) to the script(s) to be executed")
    parser.add_argument("--remote_path", type=str, help="Remote path where the script will be copied")
    parser.add_argument("--machines", type=str, help="File containing the list of machines")
    parser.add_argument("--cuda", action="store_true", help="CUDA option to be passed to the script (e.g., cuda=true or cuda=false)")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup option to be passed to the script (e.g., cleanup)")

    args = parser.parse_args()

    results = deploy_and_run_script(args.scripts, args.remote_path, args.machines, args.cuda, args.cleanup)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "results.txt")

    save_results(results, output_file)

if __name__ == "__main__":
    main()
