"""
Run all gen_data visualizations in sequence.
Output saved to: visualizations/output/

Usage:
    C:/Users/salma/anaconda3/python.exe visualizations/run_all.py
"""
import subprocess, sys, os

scripts = [
    "cora_viz.py",
    "pubmed_viz.py",
    "arxiv_viz.py",
    "wikics_viz.py",
    "kg_viz.py",
    "chemmol_viz.py",
]

folder = os.path.dirname(__file__)
python = sys.executable

for script in scripts:
    path = os.path.join(folder, script)
    print(f"\n{'='*50}")
    print(f"Running {script}...")
    print('='*50)
    result = subprocess.run([python, path], cwd=folder)
    if result.returncode != 0:
        print(f"  WARNING: {script} failed with code {result.returncode}")

print("\nAll done. Check visualizations/output/ for PNG files.")
