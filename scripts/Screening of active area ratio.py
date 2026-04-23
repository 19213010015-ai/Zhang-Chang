import os
import numpy as np
import pandas as pd
from discretisedfield import Field

# =========================
# User-defined parameters
# =========================
ANGLE_THRESHOLD_DEG = 1.0
ANGLE_THRESHOLD_RAD = np.deg2rad(ANGLE_THRESHOLD_DEG)

base_dir = os.getcwd()

# Find all folders matching the pattern R*.out
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
    # 1. Read all OVF files once
    # =========================
    all_unit_vectors = []
    mesh_shape = None

    for i, fname in enumerate(ovf_files):
        file_path = os.path.join(ovf_folder, fname)
        field = Field.from_file(file_path)

        if mesh_shape is None:
            mesh_shape = field.mesh.n  # (nx, ny, nz)

        magnetization = field.array.reshape(-1, 3)  # shape: (N_cells, 3)
        norms = np.linalg.norm(magnetization, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)  # avoid division by zero
        unit_vectors = magnetization / norms

        all_unit_vectors.append(unit_vectors)

        if (i + 1) % 10 == 0 or (i + 1) == len(ovf_files):
            print(f"  Loaded {i + 1}/{len(ovf_files)} OVF files")

    # shape: (T, N_cells, 3)
    all_unit_vectors = np.stack(all_unit_vectors, axis=0)

    n_time, n_cells, _ = all_unit_vectors.shape
    print(f"  Total number of cells: {n_cells}")

    # =========================
    # 2. Compute the rotation angle between consecutive time steps
    # =========================
    if n_time < 2:
        print(f"  Warning: fewer than 2 time steps in {folder_name}; all cells will be treated as inactive.")
        max_angles = np.zeros(n_cells)
    else:
        vec_t0 = all_unit_vectors[:-1]   # shape: (T-1, N_cells, 3)
        vec_t1 = all_unit_vectors[1:]    # shape: (T-1, N_cells, 3)

        # Dot product for each cell between adjacent time steps
        dots = np.sum(vec_t0 * vec_t1, axis=2)  # shape: (T-1, N_cells)
        dots = np.clip(dots, -1.0, 1.0)

        step_angles = np.arccos(dots)           # shape: (T-1, N_cells)
        max_angles = np.max(step_angles, axis=0)  # shape: (N_cells,)

    # =========================
    # 3. Classify cells as active or inactive using 1°
    # =========================
    inactive_mask_flat = max_angles < ANGLE_THRESHOLD_RAD
    active_mask_flat = ~inactive_mask_flat

    inactive_indices = np.where(inactive_mask_flat)[0]
    active_indices = np.where(active_mask_flat)[0]

    n_inactive = len(inactive_indices)
    n_active = len(active_indices)

    inactive_ratio = round(n_inactive / n_cells * 100, 4)
    active_ratio = round(n_active / n_cells * 100, 4)

    # =========================
    # 4. Reshape to 3D masks
    # =========================
    inactive_mask_3d = inactive_mask_flat.reshape(*mesh_shape)
    active_mask_3d = active_mask_flat.reshape(*mesh_shape)
    max_angle_3d_deg = np.rad2deg(max_angles).reshape(*mesh_shape)

    # =========================
    # 5. Save outputs
    # =========================
    output_prefix = os.path.join(ovf_folder, folder_name)

    np.savetxt(f"{output_prefix}_inactive_indices_1deg.txt", inactive_indices, fmt="%d")
    np.savetxt(f"{output_prefix}_active_indices_1deg.txt", active_indices, fmt="%d")

    np.save(f"{output_prefix}_inactive_mask_1deg.npy", inactive_mask_3d)
    np.save(f"{output_prefix}_active_mask_1deg.npy", active_mask_3d)

    # Save the maximum rotation angle of each cell in degrees
    np.save(f"{output_prefix}_max_angle_deg.npy", max_angle_3d_deg)
    np.savetxt(f"{output_prefix}_max_angle_deg_flat.txt", np.rad2deg(max_angles), fmt="%.6f")

    sample_info = {
        "Sample Folder": folder_name,
        "Total Cells": n_cells,
        "Inactive Cells (1deg)": n_inactive,
        "Active Cells (1deg)": n_active,
        "Inactive Ratio (%) (1deg)": inactive_ratio,
        "Active Ratio (%) (1deg)": active_ratio,
        "Mean Max Angle (deg)": round(np.mean(np.rad2deg(max_angles)), 6),
        "Median Max Angle (deg)": round(np.median(np.rad2deg(max_angles)), 6),
        "Std Max Angle (deg)": round(np.std(np.rad2deg(max_angles)), 6),
    }

    summary_data.append(sample_info)
    print(f"Completed {folder_name}")

# =========================
# 6. Export summary table
# =========================
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by="Inactive Ratio (%) (1deg)", ascending=False)

    summary_path = os.path.join(base_dir, "active_inactive_summary_1deg.xlsx")
    summary_df.to_excel(summary_path, index=False)

    print(f"\nAll results have been saved to: {summary_path}")
else:
    print("\nNo valid data were found for summary export.")
