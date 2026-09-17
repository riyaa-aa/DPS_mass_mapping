import os
import healpy as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import time

# 1. Configuration
TOMO_BIN = 2
DPS_MAP_FILE = f"/home2/supranta/PosteriorSampling/denoising_diffusion_pytorch/denoising_diffusion_pytorch/results_desy3_big/DPS_healpix_maps/dps_kappa_hp_tomo{TOMO_BIN}.fits"
PAIRS_FILE = 'redmapper_cluster_pairs.csv'   # <-- match get_cluster_pairs.py output
OUTPUT_DIR = 'filaments'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_SUFFIX = "_comoving3"
# notes on suffixes:
# _comoving3: newest files with the cleaned up, non-redundant code
# _shuffled_comoving: pairs are shuffled so neither one is brighter
# _comoving: using the comoving transverse distance to set dist range for cluster pairs

GRID_MIN, GRID_MAX = -2.0, 2.0
GRID_RES = 200
GAUSSIAN_SMOOTH_KERNEL = 0.05
NUM_ROTATIONS = 10  # stacks 10 randomly-rotated copies per cluster

X_LIN = np.linspace(GRID_MIN, GRID_MAX, GRID_RES)
Y_LIN = np.linspace(GRID_MIN, GRID_MAX, GRID_RES)
X_GRID, Y_GRID = np.meshgrid(X_LIN, Y_LIN)
PIXEL_WIDTH = (GRID_MAX - GRID_MIN) / GRID_RES

ALIGN_GRID_MIN, ALIGN_GRID_MAX = -0.09, 0.09
ALIGN_GRID_RES = 200
ALIGN_X_LIN = np.linspace(ALIGN_GRID_MIN, ALIGN_GRID_MAX, ALIGN_GRID_RES)
ALIGN_Y_LIN = np.linspace(ALIGN_GRID_MIN, ALIGN_GRID_MAX, ALIGN_GRID_RES)
ALIGN_X_GRID, ALIGN_Y_GRID = np.meshgrid(ALIGN_X_LIN, ALIGN_Y_LIN)
ALIGN_PIXEL_WIDTH = (ALIGN_GRID_MAX - ALIGN_GRID_MIN) / ALIGN_GRID_RES
# tuned to the effective resolution/beam of the DPS map
ALIGN_SMOOTH_KERNEL = 0.0015

def _smooth(signal):
    return gaussian_filter(signal, sigma=GAUSSIAN_SMOOTH_KERNEL / PIXEL_WIDTH)

def _align_smooth(signal):
    # separate smoothing helper since the alignment grid has its own
    # (real-radian) pixel scale, distinct from the normalized pair grid
    return gaussian_filter(signal, sigma=ALIGN_SMOOTH_KERNEL / ALIGN_PIXEL_WIDTH)

def _theta_for_row(row):
    """Angle of the L->R cluster-cluster axis"""
    ra_L, dec_L = np.radians(row.ra_L), np.radians(row.dec_L)
    ra_R, dec_R = np.radians(row.ra_R), np.radians(row.dec_R)

    alpha_mid = (ra_L + ra_R) / 2.0
    delta_mid = (dec_L + dec_R) / 2.0

    X_prime_L = -(ra_L - alpha_mid) * np.cos(delta_mid)
    Y_prime_L = dec_L - delta_mid
    X_prime_R = -(ra_R - alpha_mid) * np.cos(delta_mid)
    Y_prime_R = dec_R - delta_mid

    return np.arctan2(Y_prime_R - Y_prime_L, X_prime_R - X_prime_L)

def _theta_to_companion(target_ra, target_dec, other_ra, other_dec):
    """
    Angle from a target cluster toward its companion, projected onto a
    tangent plane centered on the TARGET cluster itself (not the pair
    midpoint, unlike _theta_for_row). Need this to rotate a single-cluser
    stack so "toward the companion" always points along +x.
    """
    X_prime_other = -(other_ra - target_ra) * np.cos(target_dec)
    Y_prime_other = other_dec - target_dec
    return np.arctan2(Y_prime_other, X_prime_other)

def stack_true_filament(dps_map, df):
    print("Stacking true cluster pairs (clusters + filament)...")
    stacked_signal = np.zeros((GRID_RES, GRID_RES))
    num_pairs = len(df)

    for i, row in enumerate(df.itertuples()):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{num_pairs} true pairs...")

        ra_L, dec_L = np.radians(row.ra_L), np.radians(row.dec_L)
        ra_R, dec_R = np.radians(row.ra_R), np.radians(row.dec_R)
        sep_rad = row.sep_rad

        alpha_mid = (ra_L + ra_R) / 2.0
        delta_mid = (dec_L + dec_R) / 2.0
        theta = _theta_for_row(row)

        X_scaled =  X_GRID * sep_rad
        Y_scaled = Y_GRID * sep_rad

        X_prime = X_scaled * np.cos(theta) - Y_scaled * np.sin(theta)
        Y_prime = X_scaled * np.sin(theta) + Y_scaled * np.cos(theta)

        alpha_eval = alpha_mid - X_prime / np.cos(delta_mid)
        delta_eval = delta_mid + Y_prime

        theta_hp = np.pi / 2.0 - delta_eval
        phi_hp = alpha_eval

        stacked_signal += hp.get_interp_val(dps_map, theta_hp, phi_hp)

    stacked_signal /= num_pairs
    return _smooth(stacked_signal)


def stack_cluster_null(dps_map, df, target_side='L', num_rotations=NUM_ROTATIONS, random_seed=42):
    """
    Re-stack the same cluster sample but with the cluster held fixed at its true
    stack position (-0.5,0) or (+0.5,0) and rotated by a random angle phi, drawn
    from [theta+pi/2, theta+3pi/2] so that the filament / opposing-cluster wedge
    is rotated OUT of the central region (Sec 3.3 of Tong et al. 2024).
    """
    print(f"Generating null stack for {target_side}-side cluster...")
    stacked_signal = np.zeros((GRID_RES, GRID_RES))
    num_pairs = len(df)
    rng = np.random.default_rng(random_seed)

    x_offset = -0.5 if target_side == 'L' else 0.5
    total_realizations = num_pairs * num_rotations

    for i, row in enumerate(df.itertuples()):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{num_pairs} null pairs ({target_side})...")

        if target_side == 'L':
            center_ra, center_dec = np.radians(row.ra_L), np.radians(row.dec_L)
        
        else:
            center_ra, center_dec = np.radians(row.ra_R), np.radians(row.dec_R)

        sep_rad = row.sep_rad
        theta = _theta_for_row(row)

        # shift so the cluster sits at (x_offset, 0), matching its true stack position
        X_scaled = (X_GRID - x_offset) * sep_rad
        Y_scaled = Y_GRID * sep_rad
        
        # restrict phi to [theta + pi/2, theta + 3pi/2] to rotate the filament
        # wedge and opposing cluster out of the evaluated region
        phi_offsets = rng.uniform(np.pi / 2.0, 3.0 * np.pi / 2.0, size=num_rotations)
        phi_rot = theta + phi_offsets # shape (num_rotations,)

        cos_p = np.cos(phi_rot)[:, None, None]
        sin_p = np.sin(phi_rot)[:, None, None]

        X_rot = X_scaled * cos_p - Y_scaled * sin_p # shape (num_rotations, GRID_RES, GRID_RES)
        Y_rot = X_scaled * sin_p + Y_scaled * cos_p

        alpha_eval = center_ra - X_rot / np.cos(center_dec)
        delta_eval = center_dec + Y_rot

        theta_hp = np.pi / 2.0 - delta_eval
        phi_hp = alpha_eval

        vals = hp.get_interp_val(dps_map, theta_hp.ravel(), phi_hp.ravel())
        stacked_signal += vals.reshape(num_rotations, GRID_RES, GRID_RES).sum(axis=0)

    stacked_signal /= total_realizations
    return _smooth(stacked_signal)

def stack_halo_alignment(dps_map, df, target_side='L', align=True, num_rotations=NUM_ROTATIONS, random_seed=42):
    """
    Stacks the DPS map centered directly on individual clusters,
    with no rescaling by pair separation; coords stay in real
    angular units (radians) around each cluster.
    If align=True: each pair contributes one realization, rotated
    so the real direction toward its companion cluster lies along +x.
    This tests whether the cluster's own mass distribution is elongated
    or skewed towards its neighbor (i.e. halo alignment).
    If align=False: this builds the isotropic null; the same single-cluster
    stack, but rotated by 'num_rotations' independent random angles per
    cluster.
    target_side='L': stacks the richer cluster, oriented towards the less
    rich cluster.
    target_side='R': stacks the less rich cluster, oriented towards the
    richer cluster.
    """
    label = 'aligned toward companion' if align else 'isotropic null'
    print(f"Stacking {target_side}-cluster halo, {label}:")

    stacked_signal = np.zeros((ALIGN_GRID_RES, ALIGN_GRID_RES))
    num_pairs = len(df)
    rng = np.random.default_rng(random_seed)

    n_realizations = num_pairs if align else num_pairs * num_rotations

    for i, row in enumerate(df.itertuples()):
        if i % 500 == 0 and i > 0:
            print(f"Processed {i}/{num_pairs} pairs...")

        if target_side == 'L':
            target_ra, target_dec = np.radians(row.ra_L), np.radians(row.dec_L)
            other_ra, other_dec = np.radians(row.ra_R), np.radians(row.dec_R)
        else:
            target_ra, target_dec = np.radians(row.ra_R), np.radians(row.dec_R)
            other_ra, other_dec = np.radians(row.ra_L), np.radians(row.dec_L)

        if align:
            phis = np.array([_theta_to_companion(target_ra, target_dec, other_ra, other_dec)])
        else:
            phis = rng.uniform(0.0, 2.0 * np.pi, size=num_rotations)

        cos_p = np.cos(phis)[:, None, None]
        sin_p = np.sin(phis)[:, None, None]

        X_rot = ALIGN_X_GRID * cos_p - ALIGN_Y_GRID * sin_p
        Y_rot = ALIGN_X_GRID * sin_p + ALIGN_Y_GRID * cos_p

        alpha_eval = target_ra - X_rot / np.cos(target_dec)
        delta_eval = target_dec + Y_rot

        theta_hp = np.pi/2.0 - delta_eval
        phi_hp = alpha_eval

        vals = hp.get_interp_val(dps_map, theta_hp.ravel(), phi_hp.ravel())
        stacked_signal += vals.reshape(len(phis), ALIGN_GRID_RES, ALIGN_GRID_RES).sum(axis=0)

    stacked_signal /= n_realizations
        
    # --- TEMPORARY DEBUG ---
    #center_idx = ALIGN_GRID_RES // 2
    #print(f"  [DEBUG] raw averaged center pixel (pre-smooth): {stacked_signal[center_idx, center_idx]:.6e}")

    return _align_smooth(stacked_signal)

def _save_map(data, filename, title, cbar_label, grid_min=GRID_MIN, grid_max=GRID_MAX):
    plt.figure(figsize=(8, 7))
    vmin, vmax = np.percentile(data, [1, 99.5])
    im = plt.imshow(data, cmap='viridis', origin='lower',
                     extent=[grid_min, grid_max, grid_min, grid_max],
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

def _side_by_side(true_stack, null_total, filename, title_suffix, use_symlog=False,
                  left_title = "True Stack", right_title="Null Total Stack",
                  grid_min = GRID_MIN, grid_max=GRID_MAX):
    """plotting helper: true stack vs null total, with a colorbar
    in its own dedicated axis"""
    fig, axes = plt.subplots(
        1, 3, figsize=(14.7, 6), constrained_layout = True,
        gridspec_kw = {'width_ratios': [1, 1, 0.05]}
    )
    ax0, ax1, cax = axes

    if use_symlog:
        all_vals = np.concatenate([true_stack.ravel(), null_total.ravel()])
        vmax = np.abs(all_vals).max()
        linthresh = 1e-3
        norm = mcolors.SymLogNorm(linthresh = linthresh, linscale=0.1, vmin=-vmax, vmax=vmax, base=10)
        imshow_kwargs = dict(norm = norm)
        cbar_label = r'DPS $\kappa$ (SymLog Scale)'
    else:
        all_vals = np.concatenate([true_stack.ravel(), null_total.ravel()])
        vmin, vmax = np.percentile(all_vals, [1, 99.5])
        imshow_kwargs = dict(vmin=vmin, vmax=vmax)
        cbar_label = r'DPS $\kappa$ Signal'

    im0 = ax0.imshow(true_stack, cmap='viridis', origin='lower',
                     extent=[grid_min, grid_max, grid_min, grid_max],
                     **imshow_kwargs)
    ax0.set_title(f'{left_title} {title_suffix}', fontsize=14)
    ax0.set_xlabel('x', fontsize=12)
    ax0.set_ylabel('y', fontsize=12)

    im1 = ax1.imshow(null_total, cmap='viridis', origin='lower',
                     extent=[grid_min, grid_max, grid_min, grid_max],
                     **imshow_kwargs)
    ax1.set_title(f'{right_title} {title_suffix}', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)

    cbar = fig.colorbar(im1, cax=cax)
    cbar.set_label(cbar_label, fontsize=12)

    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved to {filename}")

def save_null_triplet(null_left, null_right, null_total):
    """Figure-3-style panel: left null, right null, and their sum."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), constrained_layout=True,
                             gridspec_kw={'width_ratios':[1,1,1,0.05]})
    panels = [
        (null_left, 'Left cluster null'),
        (null_right, 'Right cluster null'),
        (null_total, 'Sum'),
    ]

    all_vals = np.concatenate([p[0].ravel() for p in panels])
    vmin, vmax = np.percentile(all_vals, [1, 99.5])

    im = None

    for ax, (data, title) in zip(axes[:3], panels):
        im = ax.imshow(data, cmap='viridis', origin='lower',
                        extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX],
                        vmin=vmin, vmax=vmax)
        ax.set_xlabel('x', fontsize=13)
        ax.set_title(title, fontsize=13)
    axes[0].set_ylabel('y', fontsize=13)

    cbar = fig.colorbar(im, cax=axes[3])
    cbar.set_label(r'DPS $\kappa$ Signal', fontsize=12)

    fname = os.path.join(OUTPUT_DIR, f'dps_null_maps_tomo{TOMO_BIN}{FILE_SUFFIX}.png')
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved to {fname}")

def get_mirrored_nulls(true_stack):
    """
    Build null (no-filament) models for each cluster by reflecting its outer
    half (the side away from the other cluster/filament) across its own
    center. The reflection is capped at the pair midpoint (x=0) so the left
    and right null models meet exactly there without overlapping or
    double-counting.
    """
    mid_idx   = np.argmin(np.abs(X_LIN - 0.0))
    left_idx  = np.argmin(np.abs(X_LIN - (-0.5)))
    right_idx = np.argmin(np.abs(X_LIN - 0.5))

    left_null  = np.zeros_like(true_stack)
    right_null = np.zeros_like(true_stack)

    # LEFT CLUSTER
    # Outer half (x <= -0.5): assumed uncontaminated, copied as-is
    left_null[:, :left_idx + 1] = true_stack[:, :left_idx + 1]

    # Reflect across the cluster center, but only out to x=0 -- beyond that
    # is the right cluster's / filament's territory, not the left cluster's
    inner_width_L = mid_idx - left_idx
    outer_slice_L = true_stack[:, left_idx + 1 - inner_width_L: left_idx + 1]
    left_null[:, left_idx + 1: left_idx + 1 + inner_width_L] = np.fliplr(outer_slice_L)

    # RIGHT CLUSTER
    right_null[:, right_idx:] = true_stack[:, right_idx:]

    inner_width_R = right_idx - mid_idx
    outer_slice_R = true_stack[:, right_idx: right_idx + inner_width_R]
    right_null[:, right_idx - inner_width_R: right_idx] = np.fliplr(outer_slice_R)

    return left_null, right_null

def main():

    start_time = time.perf_counter()

    df = pd.read_csv(PAIRS_FILE)
    print("Loading DPS map...")
    dps_map = hp.read_map(DPS_MAP_FILE, verbose=False)
    dps_map = dps_map - np.mean(dps_map)

    true_stack = stack_true_filament(dps_map, df)

    # mirror method to save null maps
    left_null_mirror, right_null_mirror = get_mirrored_nulls(true_stack)
    total_null_mirror = left_null_mirror + right_null_mirror
    filament_map_mirror = true_stack - total_null_mirror

    np.save(os.path.join(OUTPUT_DIR, f'filament_map_mirrored_tomo{TOMO_BIN}{FILE_SUFFIX}.npy'), filament_map_mirror)
    _save_map(filament_map_mirror,
              os.path.join(OUTPUT_DIR, f'dps_filament_mirrored_tomo{TOMO_BIN}{FILE_SUFFIX}.png'),
              f'Mirrored Subtraction Mass Map (Tomo Bin {TOMO_BIN})',
              r'DPS $\kappa$ Signal')

    np.save(os.path.join(OUTPUT_DIR, f'true_stack_tomo{TOMO_BIN}{FILE_SUFFIX}.npy'), true_stack)
    _save_map(true_stack,
               os.path.join(OUTPUT_DIR, f'dps_true_stack_tomo{TOMO_BIN}{FILE_SUFFIX}.png'),
               f'Stacked DPS Reconstruction (Tomo Bin {TOMO_BIN})',
               r'DPS $\kappa$ Signal')

    # Build null maps for left and right clusters separately (Figure 3), then sum
    null_left = stack_cluster_null(dps_map, df, target_side='L')
    null_right = stack_cluster_null(dps_map, df, target_side='R')
    null_total = null_left + null_right

    np.save(os.path.join(OUTPUT_DIR, f'null_left_tomo{TOMO_BIN}{FILE_SUFFIX}.npy'), null_left)
    np.save(os.path.join(OUTPUT_DIR, f'null_right_tomo{TOMO_BIN}{FILE_SUFFIX}.npy'), null_right)
    np.save(os.path.join(OUTPUT_DIR, f'null_total_tomo{TOMO_BIN}{FILE_SUFFIX}.npy'), null_total)
    
    print("Generating clean side-by-side comparison plot...")
    
    _side_by_side(true_stack, null_total,
                  os.path.join(OUTPUT_DIR, f'dps_comparison_linear_tomo{TOMO_BIN}{FILE_SUFFIX}.png'),
                  title_suffix=f'Tomo Bin {TOMO_BIN}', use_symlog=False)
    _side_by_side(true_stack, null_total,
                  os.path.join(OUTPUT_DIR, f'dps_comparison_symlog_tomo{TOMO_BIN}{FILE_SUFFIX}.png'),
                  title_suffix=f'Tomo Bin {TOMO_BIN}', use_symlog=True)
    _side_by_side(true_stack, total_null_mirror,
                  os.path.join(OUTPUT_DIR, f'dps_comparison_mirrored_null_linear_tomo{TOMO_BIN}{FILE_SUFFIX}.png'),
                  title_suffix=f'Tomo Bin {TOMO_BIN}, Mirrored Null Map', use_symlog=False)


    _save_map(null_left,
               os.path.join(OUTPUT_DIR, f'dps_null_left_tomo{TOMO_BIN}{FILE_SUFFIX}.png'),
               f'Null Mass Map - Left Cluster (Tomo Bin {TOMO_BIN})',
               r'DPS $\kappa$ Signal')
    _save_map(null_right,
               os.path.join(OUTPUT_DIR, f'dps_null_right_tomo{TOMO_BIN}{FILE_SUFFIX}.png'),
               f'Null Mass Map - Right Cluster (Tomo Bin {TOMO_BIN})',
               r'DPS $\kappa$ Signal')
    _save_map(null_total,
               os.path.join(OUTPUT_DIR, f'dps_null_total_tomo{TOMO_BIN}{FILE_SUFFIX}.png'),
               f'Null Mass Map - Sum (Tomo Bin {TOMO_BIN})',
               r'DPS $\kappa$ Signal')
    save_null_triplet(null_left, null_right, null_total)

    # Subtract null baseline from the true stack to isolate the filament (Figure 4)
    filament_map = true_stack - null_total
    np.save(os.path.join(OUTPUT_DIR, f'filament_map_tomo{TOMO_BIN}{FILE_SUFFIX}.npy'), filament_map)

    print("Plotting final subtracted mass map...")
    plt.figure(figsize=(8, 7))
    vmin, vmax = np.percentile(filament_map, [1, 99.5])
    im = plt.imshow(filament_map, cmap='viridis', origin='lower',
                     extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX],
                     vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label(r'DPS $\kappa$ Signal (subtracted)', fontsize=12)

    rect = plt.Rectangle((-0.3, -0.15), 0.6, 0.3, linewidth=1.5, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)

    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(f'Subtracted Mass Map (Tomo Bin {TOMO_BIN})', fontsize=14)

    output_image = os.path.join(OUTPUT_DIR, f'dps_filament_subtracted_tomo{TOMO_BIN}{FILE_SUFFIX}.png')
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Saved to {output_image}")
    
    # halo-alignment test
    align_L = stack_halo_alignment(dps_map, df, target_side='L', align=True)
    null_align_L = stack_halo_alignment(dps_map, df, target_side='L', align=False)
    excess_align_L = align_L - null_align_L

    align_R = stack_halo_alignment(dps_map, df, target_side='R', align=True)
    null_align_R = stack_halo_alignment(dps_map, df, target_side='R', align=False)
    excess_align_R = align_R - null_align_R

    for arr, name in [(align_L, f'halo_align_L_tomo{TOMO_BIN}{FILE_SUFFIX}'),
                      (null_align_L, f'halo_align_null_L_tomo{TOMO_BIN}{FILE_SUFFIX}'),
                      (excess_align_L, f'halo_align_excess_L_tomo{TOMO_BIN}{FILE_SUFFIX}'),
                      (align_R, f'halo_align_R_tomo{TOMO_BIN}{FILE_SUFFIX}'),
                      (null_align_R, f'halo_align_null_R_tomo{TOMO_BIN}{FILE_SUFFIX}'),
                      (excess_align_R, f'halo_align_excess_R_tomo{TOMO_BIN}{FILE_SUFFIX}')]:
        np.save(os.path.join(OUTPUT_DIR, f'{name}.npy'), arr)

    print("Plotting halo-alignment maps...")
    _side_by_side(align_L, null_align_L,
                  os.path.join(OUTPUT_DIR, f'dps_halo_align_comparison_L_tomo{TOMO_BIN}{FILE_SUFFIX}.png'),
                  title_suffix=f'(Richer Cluster, Tomo Bin {TOMO_BIN})', use_symlog=False,
                  left_title='Aligned Toward Companion', right_title='Isotropic Null',
                  grid_min=ALIGN_GRID_MIN, grid_max=ALIGN_GRID_MAX)
    _side_by_side(align_R, null_align_R,
                  os.path.join(OUTPUT_DIR, f'dps_halo_align_comparison_R_tomo{TOMO_BIN}{FILE_SUFFIX}.png'),
                  title_suffix=f'(Poorer Cluster, Tomo Bin {TOMO_BIN})', use_symlog=False,
                  left_title='Aligned Toward Companion', right_title='Isotropic Null',
                  grid_min=ALIGN_GRID_MIN, grid_max=ALIGN_GRID_MAX)

    _save_map(excess_align_L,
              os.path.join(OUTPUT_DIR, f'dps_halo_align_excess_L_tomo{TOMO_BIN}{FILE_SUFFIX}.png'),
              f'Halo Alignment Excess - Richer Cluster (Tomo Bin {TOMO_BIN})',
              r'DPS $\kappa$ Signal (aligned $-$ isotropic)',
              grid_min=ALIGN_GRID_MIN, grid_max=ALIGN_GRID_MAX)
    _save_map(excess_align_R,
              os.path.join(OUTPUT_DIR, f'dps_halo_align_excess_R_tomo{TOMO_BIN}{FILE_SUFFIX}.png'),
              f'Halo Alignment Excess - Poorer Cluster (Tomo Bin {TOMO_BIN})',
              r'DPS $\kappa$ Signal (aligned $-$ isotropic)',
              grid_min=ALIGN_GRID_MIN, grid_max=ALIGN_GRID_MAX)

    elapsed_time = time.perf_counter() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")


if __name__ == "__main__":
    main()
