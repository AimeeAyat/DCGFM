"""
Build PyTorch Geometric C++ extensions from source.
Required when no pre-built wheels match the installed PyTorch nightly version.
Run once during Docker image build.
"""
import subprocess
import os
import sys

env = {
    **os.environ,
    "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
    "FORCE_CUDA": "1",
    "CUDA_HOME": "/usr/local/cuda",
    "CPATH": "/usr/local/cuda/include",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
}

packages = [
    "git+https://github.com/rusty1s/pytorch_scatter.git",
    "git+https://github.com/rusty1s/pytorch_sparse.git",
    "git+https://github.com/rusty1s/pytorch_cluster.git",
    "git+https://github.com/rusty1s/pytorch_spline_conv.git",
]

for pkg in packages:
    name = pkg.split("/")[-1].replace(".git", "")
    print(f"Building {name}...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", pkg],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  OK: {name}")
    else:
        print(f"  FAILED: {name}")
        print(result.stderr[-1000:])
        sys.exit(1)

print("All PyG extensions built successfully.")
