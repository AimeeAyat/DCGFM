"""
Patch OGB library to add weights_only=False to all torch.load calls.
Required for PyTorch >= 2.6 which changed the default to weights_only=True,
breaking deserialization of PyG Data objects stored in OGB datasets.
Run once during Docker image build.
"""
import os
import glob
import importlib

def find_ogb_path():
    import ogb
    return os.path.dirname(ogb.__file__)

def patch_file(fpath):
    with open(fpath) as f:
        lines = f.readlines()
    new_lines = []
    changed = False
    for line in lines:
        if "torch.load(" in line and "weights_only" not in line and not line.strip().startswith("#"):
            # Insert weights_only=False before the last closing paren on the line
            idx = line.rindex(")")
            line = line[:idx] + ", weights_only=False" + line[idx:]
            changed = True
        new_lines.append(line)
    if changed:
        with open(fpath, "w") as f:
            f.writelines(new_lines)
    return changed

ogb_path = find_ogb_path()
py_files = glob.glob(os.path.join(ogb_path, "**", "*.py"), recursive=True)

patched = 0
for fpath in py_files:
    if patch_file(fpath):
        print(f"Patched: {fpath}")
        patched += 1

# Clear compiled bytecode so patches take effect
pyc_files = glob.glob(os.path.join(ogb_path, "**", "*.pyc"), recursive=True)
for f in pyc_files:
    os.remove(f)

print(f"\nDone. Patched {patched} files, removed {len(pyc_files)} .pyc files.")
