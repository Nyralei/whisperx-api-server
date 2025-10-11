#!/usr/bin/env python3
"""
launcher.py
----------------------
Launcher that detects GPUs and spawns worker processes.
"""


import contextlib
import os
import sys
import json
import subprocess
import time
import pathlib
import argparse
import signal
import urllib.request
import logging
from typing import Optional, List

# Configure structured logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("launcher")

# Global state
WORKERS: List[dict] = []
PROXY_PROCESS: Optional[subprocess.Popen] = None
SHUTDOWN_REQUESTED = False


# ------------------------------
# Utilities
# ------------------------------

def detect_gpu_count() -> int:
    """Detect GPUs via torch, fallback to GPU_COUNT env, default to 0."""
    try:
        import torch
        count = torch.cuda.device_count()
        logger.info(f"Detected {count} GPU(s) via torch")
        return count
    except ImportError:
        logger.warning("torch not available, falling back to GPU_COUNT env var")
    except Exception as e:
        logger.warning(f"Error detecting GPUs via torch: {e}")

    try:
        count = int(os.environ.get("GPU_COUNT", "0"))
        logger.info(f"Using GPU_COUNT from environment: {count}")
        return count
    except ValueError as e:
        logger.error(f"Invalid GPU_COUNT environment variable: {e}")
        return 0


def spawn_worker_process(gpu_index: Optional[int], port: int) -> subprocess.Popen:
    """Spawn a single uvicorn worker bound to a specific GPU/CPU and port."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu_index is None else str(gpu_index)
    env["WHISPERX_WORKER_PORT"] = str(port)

    device_str = "CPU" if gpu_index is None else f"GPU {gpu_index}"
    logger.info(f"Spawning worker on {device_str}, port {port}")

    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "whisperx_api_server.main:create_app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--factory",
        ],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    logger.info(f"Worker spawned with PID {process.pid}")
    return process


def write_worker_config(workers: List[dict], cfg_path: pathlib.Path = pathlib.Path("/tmp/whisperx_workers.json")):
    """Write the JSON config for the proxy."""
    cfg_path.write_text(json.dumps(workers, indent=2))
    logger.info(f"Worker configuration written to: {cfg_path}")
    logger.debug(f"Worker config: {json.dumps(workers, indent=2)}")


def wait_for_worker_health(host: str, port: int, timeout: float = 600.0) -> bool:
    """Poll the /healthcheck endpoint until the worker responds with 200 OK."""
    url = f"http://{host}:{port}/healthcheck"
    start = time.time()
    logger.info(f"Waiting for worker health check at {url} (timeout: {timeout}s)")

    while time.time() - start < timeout:
        if SHUTDOWN_REQUESTED:
            logger.warning("Shutdown requested during health check")
            return False

        with contextlib.suppress(Exception):
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    elapsed = time.time() - start
                    logger.info(f"Worker healthy after {elapsed:.1f}s")
                    return True
        time.sleep(0.5)

    logger.error(f"Worker failed health check after {timeout}s")
    return False


# ------------------------------
# Shutdown helpers
# ------------------------------

def terminate_process(proc: subprocess.Popen, name: str, timeout: int = 5):
    """Terminate a process gracefully, then force kill if needed."""
    try:
        logger.info(f"Terminating {name} process (PID {proc.pid})")
        proc.terminate()
        proc.wait(timeout=timeout)
        logger.info(f"{name.capitalize()} terminated gracefully")
    except subprocess.TimeoutExpired:
        logger.warning(f"{name.capitalize()} did not terminate, sending SIGKILL")
        proc.kill()
    except Exception as e:
        logger.error(f"Error terminating {name}: {e}")


def terminate_worker(pid: int, port: int):
    """Terminate a worker by PID."""
    try:
        logger.info(f"Terminating worker PID {pid} (port {port})")
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        logger.warning(f"Worker PID {pid} already terminated")
    except Exception as e:
        logger.error(f"Error terminating worker PID {pid}: {e}")


def force_kill_worker(pid: int):
    """Force kill a worker if still alive."""
    try:
        os.kill(pid, 0)
        logger.warning(f"Worker PID {pid} still alive, sending SIGKILL")
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.error(f"Error checking worker PID {pid}: {e}")


def shutdown_workers(signum=None, frame=None):
    """Gracefully terminate all worker processes and proxy on exit."""
    global SHUTDOWN_REQUESTED
    if SHUTDOWN_REQUESTED:
        logger.warning("Shutdown already in progress")
        return

    SHUTDOWN_REQUESTED = True
    signal_name = signal.Signals(signum).name if signum else "manual"
    logger.info(f"Shutting down (signal: {signal_name})...")

    if PROXY_PROCESS:
        terminate_process(PROXY_PROCESS, "proxy")

    for worker in WORKERS:
        terminate_worker(worker["pid"], worker["port"])

    time.sleep(2)
    for worker in WORKERS:
        force_kill_worker(worker["pid"])

    logger.info("Shutdown complete")
    sys.exit(0)


# ------------------------------
# Main helpers
# ------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Spawn WhisperX workers per GPU/CPU with proxy server"
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=int(os.environ.get("WORKER_START_PORT", "9000")),
        help="First worker port (default: 9000 or WORKER_START_PORT env)"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        default=os.environ.get("FORCE_CPU", "").lower() in ("true", "1", "yes"),
        help="Ignore GPUs and spawn single CPU worker"
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=int(os.environ.get("PROXY_PORT", "8000")),
        help="Port for the proxy server (default: 8000 or PROXY_PORT env)"
    )
    parser.add_argument(
        "--worker-timeout",
        type=int,
        default=int(os.environ.get("WORKER_HEALTH_TIMEOUT", "600")),
        help="Worker health check timeout in seconds (default: 600)"
    )
    return parser.parse_args()


def get_gpu_indices(force_cpu: bool) -> List[Optional[int]]:
    if force_cpu:
        logger.info("Running in CPU mode - spawning single worker")
        return [None]
    gpu_count = detect_gpu_count()
    if gpu_count == 0:
        logger.info("No GPUs detected - falling back to CPU mode")
        return [None]
    logger.info(f"Running in GPU mode - spawning {gpu_count} worker(s)")
    return list(range(gpu_count))


def start_workers(gpu_indices: List[Optional[int]], start_port: int, timeout: int):
    processes = []
    port = start_port

    for gpu_idx in gpu_indices:
        if SHUTDOWN_REQUESTED:
            break

        try:
            process = spawn_worker_process(gpu_idx, port)
            processes.append((process, gpu_idx, port))

            if wait_for_worker_health("127.0.0.1", port, timeout=timeout):
                WORKERS.append({
                    "host": "127.0.0.1",
                    "port": port,
                    "gpu": gpu_idx,
                    "pid": process.pid
                })
                logger.info(f"✓ Worker ready: port={port}, gpu={gpu_idx}, pid={process.pid}")
            else:
                logger.error(f"✗ Worker failed health check: port={port}, gpu={gpu_idx}")
                process.terminate()

        except Exception as e:
            logger.error(f"Failed to start worker on port {port}: {e}")

        port += 1
        time.sleep(0.5)

    return processes


def start_proxy(proxy_port: int) -> subprocess.Popen:
    logger.info(f"Starting proxy server on port {proxy_port}")
    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "whisperx_api_server.tools.proxy:app",
            "--host", "0.0.0.0",
            "--port", str(proxy_port)
        ],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    logger.info(f"✓ Proxy server started: port={proxy_port}, pid={process.pid}")
    return process


def monitor_processes(processes: List[tuple]):
    logger.info("=== All services running ===")
    logger.info(f"Proxy: http://0.0.0.0:{args.proxy_port}")
    logger.info(f"Workers: {len(WORKERS)}")

    try:
        while not SHUTDOWN_REQUESTED:
            if PROXY_PROCESS and PROXY_PROCESS.poll() is not None:
                logger.error(f"Proxy process died with exit code {PROXY_PROCESS.returncode}")
                shutdown_workers()
            for process, gpu_idx, port in processes:
                if process.poll() is not None:
                    logger.error(f"Worker died: port={port}, gpu={gpu_idx}, exit_code={process.returncode}")
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        shutdown_workers()
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        shutdown_workers()


# ------------------------------
# Entry point
# ------------------------------

def main():
    global PROXY_PROCESS
    global args

    args = parse_args()
    logger.info("=== WhisperX Launcher ===")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Args: {args}")

    signal.signal(signal.SIGINT, shutdown_workers)
    signal.signal(signal.SIGTERM, shutdown_workers)

    gpu_indices = get_gpu_indices(args.force_cpu)
    processes = start_workers(gpu_indices, args.start_port, args.worker_timeout)

    if not WORKERS:
        logger.error("No workers started successfully - exiting")
        sys.exit(1)

    write_worker_config(WORKERS)
    PROXY_PROCESS = start_proxy(args.proxy_port)

    monitor_processes(processes)


if __name__ == "__main__":
    main()