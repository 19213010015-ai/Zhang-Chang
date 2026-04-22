# Screening of active area ratio
import os
import numpy as np
import pandas as pd
from discretisedfield import Field


threshold_rad = np.deg2rad(1.0)  

summary = []


for folder in [f for f in os.listdir() if os.path.isdir(f) and f.startswith('R') and f.endswith('.out')]:
    ovf_files = sorted(f for f in os.listdir(folder) if f.endswith('.ovf'))
    if not ovf_files:
        continue


    f0 = Field.from_file(os.path.join(folder, ovf_files[0]))
    n_cells = np.prod(f0.mesh.n)

   
    all_vecs = []
    for fn in ovf_files:
        f = Field.from_file(os.path.join(folder, fn))
        m = f.array.reshape(-1, 3)
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12 
        all_vecs.append(m)
    all_vecs = np.stack(all_vecs, axis=0)  

    
    dots = np.einsum('tnc,tnc->tn', all_vecs[:-1], all_vecs[1:])
    angles = np.arccos(np.clip(dots, -1, 1))  # shape: (T-1, N)

    
    above_thresh = angles >= threshold_rad  # shape: (T-1, N)

    
    if above_thresh.shape[0] >= 3:
        rolling_active = (
            above_thresh[:-2] &
            above_thresh[1:-1] &
            above_thresh[2:]
        )  # shape: (T-3+1, N)

        effective = rolling_active.any(axis=0)  # shape: (N,)
    else:
        effective = np.zeros(above_thresh.shape[1], dtype=bool)

    
    ratio = effective.sum() / n_cells * 100
    summary.append({
        'samples': folder,
        'Total cells': int(n_cells),
        'Valid cells': int(effective.sum()),
        'ratio (%)': round(ratio, 2),
    })


df = pd.DataFrame(summary).sort_values('ration (%)', ascending=False)
df.to_excel('effective_domains_1deg_cont3_summary.xlsx', index=False)
print('✅effective_ratio_percent.xlsx')
