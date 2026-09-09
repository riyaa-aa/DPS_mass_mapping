import os
import healpy as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

# 1. Configuration
TOMO_BIN = 2
DPS_MAP_FILE = f"/home2/supranta/PosteriorSampling/denoising_diffusion_pytorch/denoising_diffusion_pytorch/results_desy3_big/DPS_healpix_maps/dps_kappa_hp_tomo{TOMO_BIN}.fits"
PAIRS_FILE = 'redmapper_cluster_pairs.csv'   # <-- match get_cluster_pairs.py output
OUTPUT_DIR = 'filaments'
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_MIN, GRID_MAX = -2.0, 2.0
GRID_RES = 200
GAUSSIAN_SMOOTH_KERNEL = 0.05
NUM_ROTATIONS = 10  # paper stacks 10 randomly-rotated copies per cluster (Sec 3.3)


def _theta_for_row(row):
    """Angle of the L->R cluster-cluster axis, same convention as the true stack."""
    ra_L, dec_L = np.radians(row['ra_L']), np.radians(row['dec_L'])
    ra_R, dec_R = np.radians(row['ra_R']), np.radians(row['dec_R'])

    alpha_mid = (ra_L + ra_R) / 2.0
    delta_mid = (dec_L + dec_R) / 2.0

    X_prime_L = -(ra_L - alpha_mid) * np.cos(delta_mid)
    Y_prime_L = dec_L - delta_mid
    X_prime_R = -(ra_R - alpha_mid) * np.cos(delta_mid)
    Y_prime_R = dec_R - delta_mid

    return np.arctan2(Y_prime_R - Y_prime_L, X_prime_R - X_prime_L)


def stack_true_filament():
    print("Stacking true cluster pairs (clusters + filament)...")
    df = pd.read_csv(PAIRS_FILE)
    dps_map = hp.read_map(DPS_MAP_FILE, verbose=False)
    dps_map = dps_map - np.mean(dps_map)

    x = np.linspace(GRID_MIN, GRID_MAX, GRID_RES)
    y = np.linspace(GRID_MIN, GRID_MAX, GRID_RES)
    X_grid, Y_grid = np.meshgrid(x, y)

    stacked_signal = np.zeros((GRID_RES, GRID_RES))
    num_pairs = len(df)

    for i, row in df.iterrows():
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{num_pairs} true pairs...")

        ra_L, dec_L = np.radians(row['ra_L']), np.radians(row['dec_L'])
        ra_R, dec_R = np.radians(row['ra_R']), np.radians(row['dec_R'])
        sep_rad = row['sep_rad']

        alpha_mid = (ra_L + ra_R) / 2.0
        delta_mid = (dec_L + dec_R) / 2.0

        X_prime_L = -(ra_L - alpha_mid) * np.cos(delta_mid)
        Y_prime_L = dec_L - delta_mid
        X_prime_R = -(ra_R - alpha_mid) * np.cos(delta_mid)
        Y_prime_R = dec_R - delta_mid
        theta = np.arctan2(Y_prime_R - Y_prime_L, X_prime_R - X_prime_L)

        X_scaled = X_grid * sep_rad
        Y_scaled = Y_grid * sep_rad

        X_prime = X_scaled * np.cos(theta) - Y_scaled * np.sin(theta)
        Y_prime = X_scaled * np.sin(theta) + Y_scaled * np.cos(theta)

        alpha_eval = alpha_mid - X_prime / np.cos(delta_mid)
        delta_eval = delta_mid + Y_prime

        theta_hp = np.pi / 2.0 - delta_eval
        phi_hp = alpha_eval

        stacked_signal += hp.get_interp_val(dps_map, theta_hp, phi_hp)

    stacked_signal /= num_pairs
    pixel_width = (GRID_MAX - GRID_MIN) / GRID_RES
    return gaussian_filter(stacked_signal, sigma=GAUSSIAN_SMOOTH_KERNEL / pixel_width)


def stack_cluster_null(target_side='L', num_rotations=NUM_ROTATIONS, random_seed=42):
    """
    Re-stack the same cluster sample but with the cluster held fixed at its true
    stack position (-0.5,0) or (+0.5,0) and rotated by a random angle phi, drawn
    from [theta+pi/2, theta+3pi/2] so that the filament / opposing-cluster wedge
    is rotated OUT of the central region (Sec 3.3 of Tong et al. 2024).
    """
    print(f"Generating null stack for {target_side}-side cluster...")
    df = pd.read_csv(PAIRS_FILE)
    dps_map = hp.read_map(DPS_MAP_FILE, verbose=False)
    dps_map = dps_map - np.mean(dps_map) # norm to 0

    x = np.linspace(GRID_MIN, GRID_MAX, GRID_RES)
    y = np.linspace(GRID_MIN, GRID_MAX, GRID_RES)
    X_grid, Y_grid = np.meshgrid(x, y)

    stacked_signal = np.zeros((GRID_RES, GRID_RES))
    num_pairs = len(df)
    rng = np.random.default_rng(random_seed)

    x_offset = -0.5 if target_side == 'L' else 0.5
    total_realizations = num_pairs * num_rotations

    for i, row in df.iterrows():
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{num_pairs} null pairs ({target_side})...")

        ra_L, dec_L = np.radians(row['ra_L']), np.radians(row['dec_L'])
        ra_R, dec_R = np.radians(row['ra_R']), np.radians(row['dec_R'])

        if target_side == 'L':
            center_ra, center_dec = ra_L, dec_L
        else:
            center_ra, center_dec = ra_R, dec_R

        sep_rad = row['sep_rad']
        theta = _theta_for_row(row)

        # restrict phi to [theta + pi/2, theta + 3pi/2] to rotate the filament
        # wedge and opposing cluster out of the evaluated region
        phi_offsets = rng.uniform(np.pi / 2.0, 3.0 * np.pi / 2.0, size=num_rotations)

        for phi_off in phi_offsets:
            phi_rot = theta + phi_off

            # shift the grid so the cluster sits at (x_offset, 0), matching
            # its position in the true stack, before rotating
            X_scaled = (X_grid - x_offset) * sep_rad
            Y_scaled = Y_grid * sep_rad

            X_rot = X_scaled * np.cos(phi_rot) - Y_scaled * np.sin(phi_rot)
            Y_rot = X_scaled * np.sin(phi_rot) + Y_scaled * np.cos(phi_rot)

            alpha_eval = center_ra - X_rot / np.cos(center_dec)
            delta_eval = center_dec + Y_rot

            theta_hp = np.pi / 2.0 - delta_eval
            phi_hp = alpha_eval

            stacked_signal += hp.get_interp_val(dps_map, theta_hp, phi_hp)

    stacked_signal /= total_realizations
    pixel_width = (GRID_MAX - GRID_MIN) / GRID_RES
    return gaussian_filter(stacked_signal, sigma=GAUSSIAN_SMOOTH_KERNEL / pixel_width)


def _save_map(data, filename, title, cbar_label):
    plt.figure(figsize=(8, 7))
    vmin, vmax = np.percentile(data, [1, 99.5])
    im = plt.imshow(data, cmap='viridis', origin='lower',
                     extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX],
                     vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved to {filename}")


def save_null_triplet(null_left, null_right, null_total):
    """Figure-3-style panel: left null, right null, and their sum."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), sharey=True)
    panels = [
        (null_left, 'Left cluster null'),
        (null_right, 'Right cluster null'),
        (null_total, 'Sum'),
    ]
    vmin = min(np.percentile(p[0], 1) for p in panels)
    vmax = max(np.percentile(p[0], 99.5) for p in panels)

    for ax, (data, title) in zip(axes, panels):
        im = ax.imshow(data, cmap='viridis', origin='lower',
                        extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX],
                        vmin=vmin, vmax=vmax)
        ax.set_xlabel('x', fontsize=13)
        ax.set_title(title, fontsize=13)
    axes[0].set_ylabel('y', fontsize=13)
    fig.colorbar(im, ax=axes, pad=0.02, label=r'$\Sigma$ [$10^{12}\ M_\odot\ \text{Mpc}^{-2}$]')

    fname = os.path.join(OUTPUT_DIR, f'dps_null_maps_tomo{TOMO_BIN}_comoving.png')
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved to {fname}")


def main():
    true_stack = stack_true_filament()
    np.save(os.path.join(OUTPUT_DIR, f'true_stack_tomo{TOMO_BIN}_comoving.npy'), true_stack)
    _save_map(true_stack,
               os.path.join(OUTPUT_DIR, f'dps_true_stack_tomo{TOMO_BIN}_comoving.png'),
               f'Stacked DPS Reconstruction (Tomo Bin {TOMO_BIN})',
               r'DPS $\kappa$ Signal')

    # Build null maps for left and right clusters separately (Figure 3), then sum
    null_left = stack_cluster_null(target_side='L')
    null_right = stack_cluster_null(target_side='R')
    null_total = null_left + null_right

    np.save(os.path.join(OUTPUT_DIR, f'null_left_tomo{TOMO_BIN}_comoving.npy'), null_left)
    np.save(os.path.join(OUTPUT_DIR, f'null_right_tomo{TOMO_BIN}_comoving.npy'), null_right)
    np.save(os.path.join(OUTPUT_DIR, f'null_total_tomo{TOMO_BIN}_comoving.npy'), null_total)
    
    print("Generating clean side-by-side comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Use robust percentile clipping to maintain background variance and contrast
    all_vals = np.concatenate([true_stack.ravel(), null_total.ravel()])
    vmin, vmax = np.percentile(all_vals, [1, 99.5])

    im0 = axes[0].imshow(true_stack, cmap='viridis', origin='lower',
                     extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX],
                     vmin=vmin, vmax=vmax)
    axes[0].set_title(f'True Stack (Tomo Bin {TOMO_BIN})', fontsize=14)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)

    im1 = axes[1].imshow(null_total, cmap='viridis', origin='lower',
                     extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX],
                     vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Null Total Stack (Tomo Bin {TOMO_BIN})', fontsize=14)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)

    cbar = fig.colorbar(im1, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label('DPS $\kappa$ Signal', fontsize=12)

    output_image = os.path.join(OUTPUT_DIR, f'dps_comparison_linear_tomo{TOMO_BIN}_comoving.png')
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved successfully to {output_image}")

    print("Generating side-by-side SymLog comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    all_vals = np.concatenate([true_stack.ravel(), null_total.ravel()])
    vmax = np.abs(all_vals).max()

    # linthresh defines the linear range around zero before the log scaling kicks in
    # Adjust linthresh based on the typical noise level of the map
    linthresh = 1e-4 

    norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)

    im0 = axes[0].imshow(true_stack, cmap='viridis', origin='lower',
                     extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX],
                     norm=norm)
    axes[0].set_title(f'True Stack (Tomo Bin {TOMO_BIN})', fontsize=14)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)

    im1 = axes[1].imshow(null_total, cmap='viridis', origin='lower',
                     extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX],
                     norm=norm)
    axes[1].set_title(f'Null Total Stack (Tomo Bin {TOMO_BIN})', fontsize=14)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)

    cbar = fig.colorbar(im1, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label('DPS $\kappa$ (SymLog Scale)', fontsize=12)

    output_image = os.path.join(OUTPUT_DIR, f'dps_comparison_symlog_tomo{TOMO_BIN}_comoving.png')
    plt.savefig(output_image, dpi=300)
    print(f"SymLog comparison plot saved successfully to {output_image}")

    _save_map(null_left,
               os.path.join(OUTPUT_DIR, f'dps_null_left_tomo{TOMO_BIN}_comoving.png'),
               f'Null Mass Map - Left Cluster (Tomo Bin {TOMO_BIN})',
               r'DPS $\kappa$ Signal')
    _save_map(null_right,
               os.path.join(OUTPUT_DIR, f'dps_null_right_tomo{TOMO_BIN}_comoving.png'),
               f'Null Mass Map - Right Cluster (Tomo Bin {TOMO_BIN})',
               r'DPS $\kappa$ Signal')
    _save_map(null_total,
               os.path.join(OUTPUT_DIR, f'dps_null_total_tomo{TOMO_BIN}_comoving.png'),
               f'Null Mass Map - Sum (Tomo Bin {TOMO_BIN})',
               r'DPS $\kappa$ Signal')
    save_null_triplet(null_left, null_right, null_total)

    # Subtract null baseline from the true stack to isolate the filament (Figure 4)
    filament_map = true_stack - null_total
    np.save(os.path.join(OUTPUT_DIR, f'filament_map_tomo{TOMO_BIN}_comoving.npy'), filament_map)

    print("Plotting final subtracted mass map...")
    plt.figure(figsize=(8, 7))
    vmin, vmax = np.percentile(filament_map, [1, 99.5])
    im = plt.imshow(filament_map, cmap='viridis', origin='lower',
                     extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX],
                     vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label(r'$\Sigma$ [$10^{12}\ M_\odot\ \text{Mpc}^{-2}$]', fontsize=12)

    rect = plt.Rectangle((-0.3, -0.15), 0.6, 0.3, linewidth=1.5, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)

    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(f'Subtracted Mass Map (Tomo Bin {TOMO_BIN})', fontsize=14)

    output_image = os.path.join(OUTPUT_DIR, f'dps_filament_subtracted_tomo{TOMO_BIN}_comoving.png')
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Saved to {output_image}")


if __name__ == "__main__":
    main()
