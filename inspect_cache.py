import pickle, numpy as np, sys
try:
    import torch
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})
except: pass

path = "g:/Rabia-Salman/DCGFM/OFA/cache_data/chemhiv/ST/processed"
with open(f"{path}/texts.pkl", "rb") as f:
    texts = pickle.load(f)
print("texts type:", type(texts), "len:", len(texts))
for i, t in enumerate(texts):
    if hasattr(t, 'shape'):
        print(f"  [{i}]: array shape={t.shape} dtype={t.dtype}")
    elif isinstance(t, list):
        ftype = type(t[0]).__name__ if t else 'empty'
        fval = str(t[0])[:100] if t else ''
        print(f"  [{i}]: list len={len(t)} elem_type={ftype}")
        print(f"         first: {fval}")
    else:
        print(f"  [{i}]: {type(t).__name__} = {str(t)[:100]}")
