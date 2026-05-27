#!/usr/bin/env python3
"""
Start local services: ChromaDB + Stella embedding server (optional).
Runs ChromaDB as a local process (no Docker), using your existing ./chroma_db.
"""

import subprocess
import sys
import time
import os


def start_chroma():
    """Start ChromaDB server using the chroma CLI or Python module."""
    print("Starting ChromaDB on http://localhost:8000 ...")
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Ensure PYTHONPATH includes project root if needed
    if "PYTHONPATH" not in env or project_root not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

    # Try two methods to launch ChromaDB:
    # 1) Use the 'chroma' script from venv bin if present
    python_dir = os.path.dirname(sys.executable)
    chroma_script = os.path.join(python_dir, "chroma")

    if os.path.exists(chroma_script):
        cmd = [
            chroma_script,
            "run",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--path",
            "./chroma_db",
        ]
    else:
        # 2) Fallback: python -m chromadb.cli.cli run ...
        cmd = [
            sys.executable,
            "-m",
            "chromadb.cli.cli",
            "run",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--path",
            "./chroma_db",
        ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=project_root,
    )
    time.sleep(3)
    if proc.poll() is None:
        print(f"  ✓ ChromaDB started (PID: {proc.pid})")
        return proc
    else:
        out, err = proc.communicate()
        print(f"  ✗ ChromaDB failed to start.")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  stdout: {out.decode()[:500]}")
        print(f"  stderr: {err.decode()[:500]}")
        return None


def start_stella_server():
    """Start Stella embedding server (ONNX, lazy-loads model on first request)."""
    print("Starting Stella embedding server on http://localhost:8081 ...")
    env = os.environ.copy()
    env["PORT"] = "8081"  # Set port for Stella server
    project_root = os.path.dirname(os.path.abspath(__file__))
    pythonpath = project_root
    if "PYTHONPATH" in env:
        pythonpath = project_root + ":" + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    proc = subprocess.Popen(
        [sys.executable, "stella_embedding_server/server.py"],
        env=env,
        cwd=project_root,
    )
    time.sleep(2)
    if proc.poll() is None:
        print(f"  ✓ Stella embedding server started (PID: {proc.pid})")
        print("     (Model will lazy-load on first request)")
        return proc
    else:
        out, err = proc.communicate()
        print(f"  ✗ Stella failed: {err.decode()[:500]}")
        return None


def main():
    print("=== SOTA RAG — Local Services (non-Docker) ===\n")

    import argparse

    parser = argparse.ArgumentParser(
        description="Start local ChromaDB and optional Stella server"
    )
    parser.add_argument(
        "--stella", action="store_true", help="Also start Stella embedding server"
    )
    args = parser.parse_args()

    processes = []

    # Start ChromaDB
    chroma_proc = start_chroma()
    if chroma_proc:
        processes.append(("ChromaDB", chroma_proc))
    else:
        print("\nFailed to start ChromaDB. Exiting.")
        sys.exit(1)

    # Optionally start Stella
    if args.stella:
        stella_proc = start_stella_server()
        if stella_proc:
            processes.append(("Stella", stella_proc))
        else:
            print("\nFailed to start Stella server. Continuing without it.")
    else:
        print("\nStella server not started (use --stella to enable)")

    print("\nAll services up. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n✗ {name} process exited unexpectedly.")
                    sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nShutting down services...")
        for name, proc in processes:
            proc.terminate()
            proc.wait()
            print(f"  ✓ {name} stopped")
        print("All stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
