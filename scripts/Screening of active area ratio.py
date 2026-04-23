import os
import numpy as np
import pandas as pd
from discretisedfield import Field

# =========================
# Parameter settings
# =========================
ANGLE_THRESHOLD_DEG = 1.0
ANGLE_THRESHOLD_RAD = np.deg2rad(ANGLE_THRESHOLD_DEG)

base_dir = os.getcwd()

# Find all R*.out folders in the current directory
folder_list = sorted([
    f for f in os.listdir(base_dir)
    if os.path.isdir(f) and f.startswith('R') and f.endswith('.out')
])

print(f"Found {len(folder_list)} data folders: {folder_list}")

summary_data = []

for folder_name in folder_list:
    ovf_folder = os.path.join(base_dir, folder_name)
    ovf_files = sorted([f for f in os.listdir(ovf_folder) if f.endswith('.ovf')])

    if not ovf_files:
        print(f"Skipping {folder_name}: no .ovf files found.")
        continue

    print(f"\nProcessing {folder_name} with {len(ovf_files)} time steps")

    # =========================
    # 1. Read all time-step data at once
    # =========================
    all_unit_vecs = []
    mesh_shape = None

    for i, fname in enumerate(ovf_files):
        file_path = os.path.join(ovf_folder, fname)
        f = Field.from_file(file_path)

        if mesh_shape is None:
            mesh_shape = f.mesh.n  # (nx, ny, nz)

        m = f.array.reshape(-1, 3)  # (N_cells, 3)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)  # avoid division by zero
        unit_m = m / norms

        all_unit_vecs.append(unit_m)

        if (i + 1) % 10 == 0 or (i + 1) == len(ovf_files):
            print(f"  Loaded {i + 1}/{len(ovf_files)} OVF files")

    # shape: (T, N_cells, 3)
    all_unit_vecs = np.stack(all_unit_vecs, axis=0)

    T, N_cells, _ = all_unit_vecs.shape
    print(f"  Total number of cells: {N_cells}")

    # =========================
    # 2. Calculate the rotation angle of each cell between adjacent time steps
    # =========================
    if T < 2:
        print(f"Warning: {folder_name} has fewer than 2 time steps; all cells will be treated as inactive.")
        max_angles = np.zeros(N_cells)
    else:
        vec_t0 = all_unit_vecs[:-1]   # (T-1, N_cells, 3)
        vec_t1 = all_unit_vecs[1:]    # (T-1, N_cells, 3)

        # Compute dot products for each time step and each cell
        dots = np.sum(vec_t0 * vec_t1, axis=2)  # (T-1, N_cells)
        dots = np.clip(dots, -1.0, 1.0)

        step_angles = np.arccos(dots)           # (T-1, N_cells)
        max_angles = np.max(step_angles, axis=0)  # (N_cells,)

    # =========================
    # 3. Classify active / inactive cells using a 1° threshold
    # =========================
    inactive_mask_flat = max_angles < ANGLE_THRESHOLD_RAD
    active_mask_flat = ~inactive_mask_flat

    inactive_indices = np.where(inactive_mask_flat)[0]
    active_indices = np.where(active_mask_flat)[0]

    inactive_cells = len(inactive_indices)
    active_cells = len(active_indices)

    inactive_ratio = round(inactive_cells / N_cells * 100, 4)
    active_ratio = round(active_cells / N_cells * 100, 4)

    # =========================
    # 4. Reshape to 3D masks
    # =========================
    inactive_mask_3d = inactive_mask_flat.reshape(*mesh_shape)
    active_mask_3d = active_mask_flat.reshape(*mesh_shape)
    max_angle_3d_deg = np.rad2deg(max_angles).reshape(*mesh_shape)

    # =========================
    # 5. Save results
    # =========================
    out_prefix = os.path.join(ovf_folder, folder_name)

    np.savetxt(f"{out_prefix}_inactive_indices_1deg.txt", inactive_indices, fmt="%d")
    np.savetxt(f"{out_prefix}_active_indices_1deg.txt", active_indices, fmt="%d")

    np.save(f"{out_prefix}_inactive_mask_1deg.npy", inactive_mask_3d)
    np.save(f"{out_prefix}_active_mask_1deg.npy", active_mask_3d)

    # Save the maximum rotation angle of each cell (unit: degree)
    np.save(f"{out_prefix}_max_angle_deg.npy", max_angle_3d_deg)

    # Optional: save the maximum angle in plain text format
    np.savetxt(f"{out_prefix}_max_angle_deg_flat.txt", np.rad2deg(max_angles), fmt="%.6f")

    sample_info = {
        "Sample Folder": folder_name,
        "Total Cells": N_cells,
        "Inactive Cells (1°)": inactive_cells,
        "Active Cells (1°)": active_cells,
        "Inactive Ratio (%) (1°)": inactive_ratio,
        "Active Ratio (%) (1°)": active_ratio,
        "Max Angle Mean (deg)": round(np.mean(np.rad2deg(max_angles)), 6),
        "Max Angle Median (deg)": round(np.median(np.rad2deg(max_angles)), 6),
        "Max Angle Std (deg)": round(np.std(np.rad2deg(max_angles)), 6),
    }

    summary_data.append(sample_info)
    print(f"Completed: {folder_name}")

# =========================
# 6. Export summary
# =========================
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by="Inactive Ratio (%) (1°)", ascending=False)

    summary_path = os.path.join(base_dir, "active_inactive_summary_1deg.xlsx")
    summary_df.to_excel(summary_path, index=False)

    print(f"\nAll results have been summarized in: {summary_path}")
else:
    print("\nNo valid data were available for summary export.")
