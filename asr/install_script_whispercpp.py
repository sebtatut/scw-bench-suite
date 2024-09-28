#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
import argparse
import time

def run_command(command: str, cwd=None) -> int:
    try:
        print(f"[INSTALL] Running command: {' '.join(command)}")
        result = subprocess.run(command, cwd=cwd, check=True, text=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        # get only the last 30 lines of the error message
        trimmed_error = '\n'.join(e.stderr.strip().splitlines()[-30:]) if e.stderr else "N/A"
        print(f"[INSTALL] Error: {trimmed_error}")
        partial_cleanup(cwd)
        sys.exit(1)

def check_cuda() -> bool:
    try:
        print("[INSTALL] Checking if GPU and CUDA toolkit...")
        subprocess.run("nvidia-smi", cwd=None, check=True, text=True)
        print("[INSTALL] Has GPU")
        subprocess.run(["nvcc", "--version"], cwd=None, check=True, text=True)
        print("[INSTALL] Has CUDA toolkit")
        return True
    except Exception:
        print("[INSTALL] No GPU or CUDA toolkit found")
        return False

def install_dependencies() -> None:
    print("[INSTALL] Updating and upgrading system packages...")
    run_command(["sudo", "apt", "update"])
    # run_command(["sudo", "apt", "upgrade", "-y"])
    
    print("[INSTALL] Installing necessary packages...")
    run_command(["sudo", "apt", "install", "ccache", "pkg-config", "ffmpeg", "build-essential", "make", "gcc", "-y"])

def clone_repositories() -> None:
    print("[INSTALL] Cloning repositories...")
    os.makedirs(os.path.expanduser("~/repos"), exist_ok=True)
    run_command(["git", "clone", "https://github.com/OpenMathLib/OpenBLAS.git"], cwd=os.path.expanduser("~/repos"))
    run_command(["git", "clone", "https://github.com/ggerganov/whisper.cpp.git"], cwd=os.path.expanduser("~/repos"))

def download_models() -> None:
    print("[INSTALL] Downloading models...")
    os.makedirs(os.path.expanduser("~/models"), exist_ok=True)
    model_urls = [
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
    ]
    for url in model_urls:
        run_command(["wget", url], cwd=os.path.expanduser("~/models"))

def install_openblas() -> None:
    print("[INSTALL] Building and installing OpenBLAS...")
    openblas_dir = os.path.expanduser("~/repos/OpenBLAS")
    run_command(["make", "clean"], cwd=openblas_dir)
    run_command(["make", "C_LAPACK=1", "NO_FORTRAN=1", "NO_LAPACK=1"], cwd=openblas_dir)
    run_command(["sudo", "make", "PREFIX=/usr/local", "C_LAPACK=1", "NO_FORTRAN=1", "NO_LAPACK=1", "install"], cwd=openblas_dir)
    # run echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/openblas.conf
    run_command(["echo", "/usr/local/lib", "|", "sudo", "tee", "/etc/ld.so.conf.d/openblas.conf"])
    run_command(["sudo", "ldconfig"])

def install_whispercpp(cuda: bool) -> None:
    print("[INSTALL] Building and installing whisper.cpp...")
    whispercpp_dir = os.path.expanduser("~/repos/whisper.cpp")
    run_command(["make", "clean"], cwd=whispercpp_dir)

    cuda_flag = "0"

    # check if CUDA is available
    if cuda and check_cuda(): 
        cuda_flag = "1"
        run_command(["make", "-j", f"GGML_OPENBLAS=1", f"GGML_CUDA={cuda_flag}"], cwd=whispercpp_dir)
    else:
        run_command(["make", "-j", "GGML_OPENBLAS=1"], cwd=whispercpp_dir)
    run_command(["make", "bench"], cwd=whispercpp_dir)
    run_command(["make", "samples"], cwd=whispercpp_dir)

def partial_cleanup(cwd: str) -> None:
    print("[INSTALL] Cleaning up after error...")
    if cwd:
        run_command(["make", "clean"], cwd=cwd)
    print("[INSTALL] Partial cleanup completed. Repositories and models are preserved.")

def full_cleanup() -> None:
    print("[INSTALL] Cleaning up all installation files...")
    shutil.rmtree(os.path.expanduser("~/repos"), ignore_errors=True)
    shutil.rmtree(os.path.expanduser("~/models"), ignore_errors=True)
    print("[INSTALL] Full cleanup completed.")

def install_all(cuda: bool) -> None:
    install_dependencies()
    clone_repositories()
    download_models()
    install_openblas()
    install_whispercpp(cuda)

def main():
    parser = argparse.ArgumentParser(description="Installation script for Whisper.cpp")
    parser.add_argument("command", type=str, choices=["install", "cleanup"], help="Command to execute: install or cleanup")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA support")

    args = parser.parse_args()

    if args.command == "install":
        install_all(args.cuda)
    elif args.command == "cleanup":
        full_cleanup()
    else:
        print("[INSTALL] Unknown command. Use 'install' or 'cleanup'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
