# The proportion of energy and power dissipation from different perspectives
import os, sys, glob, re, struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Energy power loss under different angle thresholds
def get_mesh_dims(file_path):
    """Extract xnodes, ynodes, znodes from OVF header using regex."""
    header = open(file_path, "rb").read(10000).decode("utf-8", errors="ignore")
    m = re.search(r"xnodes\s*[:=]\s*(\d+)", header, re.IGNORECASE)
    n = re.search(r"ynodes\s*[:=]\s*(\d+)", header, re.IGNORECASE)
    p = re.search(r"znodes\s*[:=]\s*(\d+)", header, re.IGNORECASE)
    if not (m and n and p):
        raise ValueError("Cannot parse mesh dimensions from header.")
    return int(m.group(1)), int(n.group(1)), int(p.group(1))

def read_vectors(file_path, nvec):
    """Read Binary4 OVF vectors, return (nvec,3) float32 array."""
    data = open(file_path, "rb").read()
    key = b"# Begin: Data Binary 4"
    idx = data.find(key)
    if idx < 0:
        raise ValueError(f"No Binary4 section in {file_path}")
    start = data.find(b"\n", idx) + 1
    raw = data[start:start + nvec*3*4]
    vals = struct.unpack(f"<{nvec*3}f", raw)
    return np.array(vals, dtype=np.float32).reshape(-1, 3)

def compute_delta_angles(prev, curr):
    """Compute per-cell rotation angle between prev and curr vectors."""
    dot = np.einsum("ij,ij->i", prev, curr)
    nr = np.linalg.norm(prev, axis=1)
    nc = np.linalg.norm(curr, axis=1)
    denom = nr * nc
    cosθ = np.zeros_like(dot)
    valid = denom > 1e-12
    cosθ[valid] = dot[valid] / denom[valid]
    cosθ = np.clip(cosθ, -1.0, 1.0)
    angles = np.zeros_like(dot)
    angles[valid] = np.degrees(np.arccos(cosθ[valid]))
    return angles, nc

def main(ovf_dir, thresholds=None):
    if thresholds is None:
        thresholds = [0,0.5,1,1.5,2,2.5,3]

    files = sorted(glob.glob(os.path.join(ovf_dir, "*.ovf")))
    if len(files) < 2:
        print("Need at least two OVF files for dynamic analysis.")
        return

    # Mesh dims and number of vectors
    xnodes, ynodes, znodes = get_mesh_dims(files[0])
    nvec = xnodes * ynodes * znodes

    # Initialize accumulators
    total_diss = 0.0
    diss_by_th = {th: 0.0 for th in thresholds}
    count_by_th = {th: 0 for th in thresholds}

    # Read first frame
    prev = read_vectors(files[0], nvec)

    # Loop over consecutive pairs
    for f in files[1:]:
        curr = read_vectors(f, nvec)
        angles, mags = compute_delta_angles(prev, curr)
        step_diss = mags * np.sin(np.radians(angles))
        total_diss += step_diss.sum()

        for th in thresholds:
            mask = angles > th
            diss_by_th[th] += step_diss[mask].sum()
            count_by_th[th] += np.count_nonzero(mask)

        prev = curr  # update reference

    # Prepare results DataFrame
    df = pd.DataFrame({
        "Threshold (°)": thresholds,
        "Effective Count": [count_by_th[th] for th in thresholds],
        "Dissipation Proxy": [diss_by_th[th] for th in thresholds],
        "Dissipation Ratio (%)": [
            (diss_by_th[th] / total_diss * 100) if total_diss>0 else 0.0
            for th in thresholds
        ]
    })

    # Print to console
    print("\n=== Cumulative Dissipation Analysis ===")
    print(df.to_string(index=False))

    # Save to CSV
    out_csv = os.path.join(ovf_dir, "dissipation_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")

    # Plot Dissipation Ratio vs Threshold
    plt.figure(figsize=(8,5))
    plt.plot(df["Threshold (°)"], df["Dissipation Ratio (%)"], marker='o')
    plt.xlabel("Rotation Angle Threshold (°)")
    plt.ylabel("Cumulative Dissipation Ratio (%)")
    plt.title("Energy Dissipation Ratio vs Angle Threshold")
    plt.grid(True)
    plt.tight_layout()

    # Save and show plot
    plot_path = os.path.join(ovf_dir, "dissipation_ratio_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    main(directory)

