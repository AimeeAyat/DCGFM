import json, os

dirs = [
    'F:/Rabia-Salman/DCGFM/OFA/saved_exp',
    'F:/Rabia-Salman/DCGFM/OFA/saved_exp_old'
]

rows = []
for d in dirs:
    for exp in sorted(os.listdir(d)):
        rpath = os.path.join(d, exp, 'results.json')
        cpath = os.path.join(d, exp, 'command')
        if not os.path.exists(rpath):
            continue
        with open(rpath) as f:
            r = json.load(f)
        hiv_keys   = [k for k in r if 'chemhiv'  in k and 'loss' not in k]
        pcba_keys  = [k for k in r if 'chempcba' in k]
        arxiv_keys = [k for k in r if 'arxiv'    in k and 'loss' not in k]
        if not hiv_keys:
            continue
        hiv  = r[hiv_keys[0]]
        pcba = r[pcba_keys[0]] if pcba_keys else None
        has_full = bool(arxiv_keys)
        rev = 'unknown'
        if os.path.exists(cpath):
            for line in open(cpath, encoding='utf-8', errors='ignore'):
                if 'hard_pruning_reverse' in line:
                    rev = line.strip().split()[-1]
                    break
        rows.append((exp, rev, hiv['mean'], hiv['std'],
                     pcba['mean'] if pcba else None,
                     pcba['std']  if pcba else None,
                     has_full))

def pruning_label(rev):
    return 'near-center' if rev == 'true' else 'far-center'

def fmt(m, s):
    return f'{m:.4f} +- {s:.4f}' if m is not None else '-'

print('=== MOL-ONLY EXPERIMENTS ===')
print(f'{"Experiment":<60} {"Pruning":<12} {"chemhiv AUC":<22} {"chempcba AUCmulti"}')
print('-' * 115)
for name, rev, hm, hs, pm, ps, full in rows:
    if full:
        continue
    print(f'{name[:59]:<60} {pruning_label(rev):<12} {fmt(hm,hs):<22} {fmt(pm,ps)}')

print()
print('=== FULL DATASET EXPERIMENTS ===')
print(f'{"Experiment":<60} {"Pruning":<12} {"chemhiv AUC":<22} {"chempcba AUCmulti"}')
print('-' * 115)
for name, rev, hm, hs, pm, ps, full in rows:
    if not full:
        continue
    print(f'{name[:59]:<60} {pruning_label(rev):<12} {fmt(hm,hs):<22} {fmt(pm,ps)}')
