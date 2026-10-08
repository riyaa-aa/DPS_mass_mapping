import os
import healpy as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
import time
import sys
from multiprocessing import Pool

# 1. Configuration
#TOMO_BIN = 4
TOMO_BIN = int(os.environ.get("TOMO_BIN", 4))
DPS_MAP_FILE = f"/home2/supranta/PosteriorSampling/denoising_diffusion_pytorch/denoising_diffusion_pytorch/results_desy3_big/DPS_healpix_maps/dps_kappa_hp_tomo{TOMO_BIN}.fits"
PAIRS_FILE = 'redmapper_cluster_pairs.csv'   # <-- match get_cluster_pairs.py output
OUTPUT_DIR = 'filaments'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_SUFFIX = "_manidk"
# notes on suffixes:
# _comoving3: newest files with the cleaned up, non-redundant code
# _shuffled_comoving: pairs are shuffled so neither one is brighter
# _comoving: using the comoving transverse distance to set dist range for cluster pairs
# rot[num]: how many NUM_ROTATIONS i use to stack the null maps
# _unsmoothed: without using the _smooth function
# _halfsplit: each cluster null only applies to its own half (instead of left + right)
# _radial_profile_null: constructing null maps using a 1d radial profile
# _companion_excluded: using exclude_companion=True in stack_cluster_null
# _sep_bg: subtract background kappa from each left and right before summing, then add one background back
#           (used in get_2halo_linear_null function)
# _2halos_isolated: using both exclude_companion and get_2halo_linear_null
# _plswork: if i have to debug anymore my eyes are going to fall out of my head
# _m0: without the quadrupole term (as a filament fit)

GRID_MIN, GRID_MAX = -2.0, 2.0
GRID_RES = 200
GAUSSIAN_SMOOTH_KERNEL = 0.05
NUM_ROTATIONS = 10  # stacks 10 randomly-rotated copies per cluster

X_LIN = np.linspace(GRID_MIN, GRID_MAX, GRID_RES)
Y_LIN = np.linspace(GRID_MIN, GRID_MAX, GRID_RES)
X_GRID, Y_GRID = np.meshgrid(X_LIN, Y_LIN)
PIXEL_WIDTH = (GRID_MAX - GRID_MIN) / (GRID_RES - 1)

ALIGN_GRID_MIN, ALIGN_GRID_MAX = -0.09, 0.09
ALIGN_GRID_RES = 200
ALIGN_X_LIN = np.linspace(ALIGN_GRID_MIN, ALIGN_GRID_MAX, ALIGN_GRID_RES)
ALIGN_Y_LIN = np.linspace(ALIGN_GRID_MIN, ALIGN_GRID_MAX, ALIGN_GRID_RES)
ALIGN_X_GRID, ALIGN_Y_GRID = np.meshgrid(ALIGN_X_LIN, ALIGN_Y_LIN)
ALIGN_PIXEL_WIDTH = (ALIGN_GRID_MAX - ALIGN_GRID_MIN) / (ALIGN_GRID_RES - 1)
ALIGN_SMOOTH_KERNEL = 0.0015

EXCLUSION_FRAC = 0.5 # exclude points within this fraction of the pair
WEDGE_HALF_ANGLE = np.radians(45.0)
WEDGE_RMIN_FRAC = 0.5
WEDGE_RMAX_FRAC = 1.5

JOINT_CONFIGS = {"Joint fit": dict(m2=False), "Joint fit + m2": dict(m2=True)}
#JOINT_CONFIGS = {
#    "Joint m2 (excl 0.5)": dict(m2=True, excl=(0.5, 0.3)),
#    "Joint m2 (excl 0.6)": dict(m2=True, excl=(0.6, 0.3)),
#}

BOX_MASK = (X_GRID >= -0.3) & (X_GRID <= 0.3) & (Y_GRID >= -0.15) & (Y_GRID <= 0.15)

_ny, _nx = GRID_RES, GRID_RES
_border_y, _border_x = int(_ny * 0.15), int(_nx * 0.15)
CORNER_MASK = np.zeros((_ny, _nx), dtype=bool)
CORNER_MASK[:_border_y, :_border_x] = True
CORNER_MASK[:_border_y, -_border_x:] = True
CORNER_MASK[-_border_y:, :_border_x] = True
CORNER_MASK[-_border_y:, -_border_x:] = True

N_SHUFFLES = 100
N_PROC = 8          # set to core count; also run with OMP_NUM_THREADS=1
_G = {}

def _smooth(signal):
    return gaussian_filter(signal, sigma=GAUSSIAN_SMOOTH_KERNEL / PIXEL_WIDTH)

def _smooth_rotations(vals):
    """
    Smooth each rotation independently.

    vals shape = (n_rotations, GRID_RES, GRID_RES)
    Set sigma=0 along the rotation axis so rotations are never
    mixed with one another.
    """
    sigma_pix = GAUSSIAN_SMOOTH_KERNEL / PIXEL_WIDTH
    return gaussian_filter(
        vals,
        sigma=(0.0, sigma_pix, sigma_pix)
    )

def _align_smooth(signal):
    # separate smoothing helper since the alignment grid has its own
    # (real-radian) pixel scale, distinct from the normalized pair grid
    return gaussian_filter(signal, sigma=ALIGN_SMOOTH_KERNEL / ALIGN_PIXEL_WIDTH)

def _theta_for_row(row):
    """
    Angle of the L->R cluster-cluster axis in the same tangent-plane
    convention used throughout the stack.

    Convention:
        +x corresponds to decreasing RA because
        alpha_eval = alpha_mid - X / cos(delta_mid)
    """
    ra_L, dec_L = np.radians(row.ra_L), np.radians(row.dec_L)
    ra_R, dec_R = np.radians(row.ra_R), np.radians(row.dec_R)

    # Wrapped L -> R RA displacement
    dra = (ra_R - ra_L + np.pi) % (2 * np.pi) - np.pi

    # Pair midpoint
    alpha_mid = (ra_L + 0.5 * dra) % (2 * np.pi)
    delta_mid = 0.5 * (dec_L + dec_R)

    # Wrapped RA offsets from midpoint
    dra_L = (ra_L - alpha_mid + np.pi) % (2 * np.pi) - np.pi
    dra_R = (ra_R - alpha_mid + np.pi) % (2 * np.pi) - np.pi

    # Same x convention as alpha_eval = alpha_mid - X/cos(delta)
    X_L = -dra_L * np.cos(delta_mid)
    Y_L = dec_L - delta_mid

    X_R = -dra_R * np.cos(delta_mid)
    Y_R = dec_R - delta_mid

    return np.arctan2(Y_R - Y_L, X_R - X_L)

def _theta_to_companion(ra1, dec1, ra2, dec2):
    """
    Angle from a target cluster toward its companion, projected onto a
    tangent plane centered on the TARGET cluster itself (not the pair
    midpoint, unlike _theta_for_row). Need this to rotate a single-cluser
    stack so "toward the companion" always points along +x.
    """
    #d_ra = (other_ra - target_ra + np.pi) % (2.0 * np.pi) - np.pi
    #X_prime_other = -d_ra * np.cos(target_dec) # fix RA overflow error
    #Y_prime_other = other_dec - target_dec
    
    #return np.arctan2(Y_prime_other, X_prime_other)
    dra = (ra2 - ra1 + np.pi) % (2*np.pi) - np.pi
    ddec = dec2 - dec1

    x_comp = -dra * np.cos(dec1)
    y_comp = ddec

    return np.arctan2(y_comp, x_comp)


def compute_bootstrap_stats(per_pair_kappas, n_bootstraps=1000, seed=42):
    """
    Computes bootstrap standard error (sigma_kappa) and 68.3% confidence intervals
    from an array of per-pair box kappa values (shape: [N_pairs]).
    """
    per_pair_kappas = np.asarray(per_pair_kappas)
    n_pairs = len(per_pair_kappas)
    rng = np.random.default_rng(seed)

    # Resample with replacement across pairs
    boot_indices = rng.choice(n_pairs, size=(n_bootstraps, n_pairs), replace=True)
    boot_means = np.mean(per_pair_kappas[boot_indices], axis=1)

    mean_val = np.mean(boot_means)
    sigma_kappa = np.std(boot_means, ddof=1)
    snr = mean_val / sigma_kappa if sigma_kappa > 0 else 0.0

    ci_lower = np.percentile(boot_means, 15.85)
    ci_upper = np.percentile(boot_means, 84.15)

    return {
        "mean": mean_val,
        "sigma": sigma_kappa,
        "snr": snr,
        "ci_68": (ci_lower, ci_upper)
    }


def stack_true_filament(dps_map, df, smooth=True, weights=None, x_range=(-0.3, 0.3)):
    print("Stacking true cluster pairs (clusters + filament)...")
    stacked_signal = np.zeros((GRID_RES, GRID_RES))
    num_pairs = len(df)
    sum_map = np.zeros((GRID_RES, GRID_RES))
    true_box_kappas = np.zeros(num_pairs)
    pair_stats = np.zeros(num_pairs)

    #x_mask = (X_GRID[0, :] >= x_range[0]) & (X_GRID[0, :] <= x_range[1])
    #per_pair_raw_profiles = np.zeros((num_pairs, GRID_RES))

    for i, row in enumerate(df.itertuples()):
        #if i % 500 == 0 and i > 0:
        #    print(f"  Processed {i}/{num_pairs} true pairs...")

        ra_L, dec_L = np.radians(row.ra_L), np.radians(row.dec_L)
        ra_R, dec_R = np.radians(row.ra_R), np.radians(row.dec_R)
        sep_rad = row.sep_rad

        #alpha_mid = (ra_L + ra_R) / 2.0
        dra = (ra_R - ra_L + np.pi) % (2*np.pi) - np.pi # solve an RA-overflow problem 
        alpha_mid = (ra_L + 0.5 * dra) % (2*np.pi)
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

        #stacked_signal += hp.get_interp_val(dps_map, theta_hp, phi_hp)
    
        pair_map = hp.get_interp_val(dps_map,theta_hp,phi_hp)

        if smooth:
            pair_map = _smooth(pair_map)
        
        if weights is not None: 
            pair_stats[i] = np.sum(weights * pair_map)

        sum_map += pair_map
        true_box_kappas[i] = np.mean(pair_map[BOX_MASK])

        #per_pair_raw_profiles[i, :] = np.mean(pair_map[:, x_mask], axis=1)
        
        if weights is not None:
            pair_stats[i] = np.sum(weights * pair_map)

    avg_stack = sum_map / num_pairs

    #stacked_signal /= num_pairs
    #return _smooth(stacked_signal)
    #return stacked_signal
    
    if weights is None:
        return avg_stack, true_box_kappas

    return avg_stack, true_box_kappas, pair_stats


def stack_cluster_null(dps_map, df, target_side='L', num_rotations=NUM_ROTATIONS, random_seed=42,
                       exclude_companion=True, exclusion_frac=EXCLUSION_FRAC, smooth_mode='after_mask',
                       use_source_wedge=False,wedge_half_angle=WEDGE_HALF_ANGLE,
                       wedge_rmin_frac=WEDGE_RMIN_FRAC,wedge_rmax_frac=WEDGE_RMAX_FRAC):
    """
    Re-stack the same cluster sample but with the cluster held fixed at its true
    stack position (-0.5,0) or (+0.5,0) and rotated by a random angle phi, drawn
    from [theta+pi/2, theta+3pi/2] so that the filament / opposing-cluster wedge
    is rotated OUT of the central region.
    """
    print(f"Generating null stack for {target_side}-side cluster...")
    stacked_signal = np.zeros((GRID_RES, GRID_RES))
    valid_counts = np.zeros((GRID_RES, GRID_RES))
    num_pairs = len(df)
    sum_map = np.zeros((GRID_RES, GRID_RES))
    null_box_kappas = np.zeros(num_pairs)
    null_bg_kappas = np.zeros(num_pairs)

    rng = np.random.default_rng(random_seed)

    x_offset = -0.5 if target_side == 'L' else 0.5
    total_realizations = num_pairs * num_rotations

    for i, row in enumerate(df.itertuples()):
        #if i % 500 == 0 and i > 0:
        #    print(f"  Processed {i}/{num_pairs} null pairs ({target_side})...")

        if target_side == 'L':
            center_ra, center_dec = np.radians(row.ra_L), np.radians(row.dec_L)
            other_ra, other_dec = np.radians(row.ra_R), np.radians(row.dec_R)
        else:
            center_ra, center_dec = np.radians(row.ra_R), np.radians(row.dec_R)
            other_ra, other_dec = np.radians(row.ra_L), np.radians(row.dec_L)

        sep_rad = row.sep_rad
        theta = _theta_for_row(row)
        companion_theta = _theta_to_companion(center_ra, center_dec, other_ra, other_dec)

        # shift so the cluster sits at (x_offset, 0), matching its true stack position
        X_scaled = (X_GRID - x_offset) * sep_rad
        Y_scaled = Y_GRID * sep_rad
        
        # restrict phi to [theta + pi/2, theta + 3pi/2] to rotate the filament
        # wedge and opposing cluster out of the evaluated region
        phi_offsets = rng.uniform(np.pi / 2.0, 3.0 * np.pi / 2.0, size=num_rotations)
        grid_companion_angle = 0.0 if target_side == 'L' else np.pi
        phi_rot = companion_theta + grid_companion_angle + phi_offsets # shape (num_rotations,)
        #phi_rot = np.array([theta + (np.pi/2)])

        cos_p = np.cos(phi_rot)[:, None, None]
        sin_p = np.sin(phi_rot)[:, None, None]

        X_rot = X_scaled * cos_p - Y_scaled * sin_p # shape (num_rotations, GRID_RES, GRID_RES)
        Y_rot = X_scaled * sin_p + Y_scaled * cos_p

        alpha_eval = center_ra - X_rot / np.cos(center_dec)
        delta_eval = center_dec + Y_rot

        theta_hp = np.pi / 2.0 - delta_eval
        phi_hp = alpha_eval

        vals = hp.get_interp_val(dps_map, theta_hp.ravel(), phi_hp.ravel())
        vals = vals.reshape(num_rotations, GRID_RES, GRID_RES)

        # Optional smoothing BEFORE masking
        if smooth_mode == 'before_mask':
            vals_work = _smooth_rotations(vals)

        elif smooth_mode in ('after_mask', 'none'):
            vals_work = vals

        else:
            raise ValueError(
                "smooth_mode must be 'after_mask', "
                "'before_mask', or 'none'"
            )

        # Companion exclusion + rotation averaging
        if exclude_companion or use_source_wedge:
            # Source position relative to companion
            dra_other = (alpha_eval - other_ra + np.pi) % (2 * np.pi) - np.pi
            x_other = -dra_other * np.cos(other_dec)
            y_other = delta_eval - other_dec
            dist_to_companion = np.sqrt(x_other**2 + y_other**2)
            # Start with every sampled source point allowed
            mask = np.ones_like(dist_to_companion, dtype=float)

            # A. Circular companion exclusion
            if exclude_companion:
                exclude_radius = exclusion_frac * sep_rad
                radial_mask = (dist_to_companion > exclude_radius)
                mask *= radial_mask.astype(float)

            # B. Source-space companion-direction wedge
            if use_source_wedge:
                # Target-centered source coordinates
                dra_target = (alpha_eval - center_ra + np.pi) % (2 * np.pi) - np.pi
                x_target = (-dra_target * np.cos(center_dec))
                y_target = (delta_eval - center_dec)
                r_target = np.sqrt(x_target**2 + y_target**2)
                source_angle = np.arctan2(y_target,x_target)

                # Angular distance from target -> companion direction
                angle_diff = (source_angle - companion_theta + np.pi) % (2 * np.pi) - np.pi
                # Radial range in units of pair separation
                rmin = wedge_rmin_frac * sep_rad
                rmax = wedge_rmax_frac * sep_rad

                inside_wedge = ((np.abs(angle_diff) < wedge_half_angle)&
                                (r_target >= rmin)&
                                (r_target <= rmax))

                # Remove source pixels in the companion-side wedge
                mask *= (~inside_wedge).astype(float)

            # Average over valid rotations
            numerator = (vals * mask).sum(axis=0)
            denominator = mask.sum(axis=0)

            pair_map = np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator > 0)
        
        else:
            pair_map = vals.mean(axis=0)

        # Optional smoothing AFTER masking
        if smooth_mode == 'after_mask':
            pair_map_smoothed = _smooth(pair_map)

        elif smooth_mode == 'before_mask':
            pair_map_smoothed = pair_map

        elif smooth_mode == 'none':
            pair_map_smoothed = pair_map

        sum_map += pair_map_smoothed

        null_box_kappas[i] = np.mean(pair_map_smoothed[BOX_MASK])
        null_bg_kappas[i] = np.mean(pair_map_smoothed[CORNER_MASK])
        #stacked_signal += vals.reshape(num_rotations, GRID_RES, GRID_RES).sum(axis=0)

    #stacked_signal /= total_realizations
    
    #stacked_signal = np.divide(
    #    stacked_signal, valid_counts,
    #    out=np.zeros_like(stacked_signal),
    #    where=valid_counts > 0
    #)

    #return _smooth(stacked_signal)
    #return stacked_signal

    avg_stack = sum_map / num_pairs
    return avg_stack, null_box_kappas, null_bg_kappas

def get_2halo_linear_null(null_left, null_right, corner_fraction=0.15):
    """
    Constructs a physically accurate 2-halo null baseline map by linearly 
    summing the isolated left and right halo profiles over a single 
    background floor.
    """
    # 1. Define mask for far-field background estimation (corners of the grid)
    # Uses top/bottom 15% of pixels where cluster signal has decayed to zero
    ny, nx = null_left.shape
    border_y = int(ny * corner_fraction)
    border_x = int(nx * corner_fraction)
    
    corner_mask = np.zeros((ny, nx), dtype=bool)
    corner_mask[:border_y, :border_x] = True
    corner_mask[:border_y, -border_x:] = True
    corner_mask[-border_y:, :border_x] = True
    corner_mask[-border_y:, -border_x:] = True

    # 2. Estimate background floor for each null realization
    bg_L = np.mean(null_left[corner_mask])
    bg_R = np.mean(null_right[corner_mask])
    bg_mean = 0.5 * (bg_L + bg_R)

    # 3. Isolate 1-halo profiles by stripping background
    halo_L = null_left - bg_L
    halo_R = null_right - bg_R

    # 4. Superimpose 1-halo profiles linearly over a single background floor
    null_2halo_total = halo_L + halo_R + bg_mean

    return null_2halo_total


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
        
    return _align_smooth(stacked_signal)

def _build_cluster_radial_null(true_stack, x_center, y_center, valid_mask, n_bins=60):
    """
    Measures kappa(r) from valid_mask pixels, builds a 1D interpolated profile,
    and projects it into a 2D isotropic null map centered on (x_center, y_center).
    """
    # 1. Compute radial distances for all grid points relative to cluster center
    R_GRID = np.sqrt((X_GRID - x_center)**2 + (Y_GRID - y_center)**2)
    
    # 2. Extract radii and kappa values only from the uncontaminated mask region
    r_vals = R_GRID[valid_mask]
    kappa_vals = true_stack[valid_mask]
    
    # 3. Bin pixels radially to compute the mean 1D profile kappa(r)
    r_max = R_GRID.max()
    bin_edges = np.linspace(0.0, r_max, n_bins + 1)
    mean_kappa, _, _ = binned_statistic(r_vals, kappa_vals, statistic='mean', bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Clean up empty bins if any exist
    valid_bins = ~np.isnan(mean_kappa)
    bin_centers = bin_centers[valid_bins]
    mean_kappa = mean_kappa[valid_bins]
    
    # 4. Create 1D interpolator (clamp bounds to prevent extrapolation errors)
    profile_interp = interp1d(
        bin_centers, 
        mean_kappa, 
        kind='linear', 
        bounds_error=False, 
        fill_value=(mean_kappa[0], mean_kappa[-1])
    )
    
    # 5. Evaluate the 1D profile across the entire 2D coordinate grid
    null_map = profile_interp(R_GRID)
    return null_map


def get_radial_profile_nulls(true_stack):
    """
    Builds standalone 2D null maps for Left and Right clusters by reconstructing
    their 1D azimuthally averaged radial profiles from their outer half-planes.
    """
    # Define cluster centers
    x_L, y_L = -0.5, 0.0
    x_R, y_R =  0.5, 0.0

    # Masks for uncontaminated outer half-planes
    mask_left  = (X_GRID <= x_L)
    mask_right = (X_GRID >= x_R)

    # Generate individual smooth 2D null profiles
    left_null  = _build_cluster_radial_null(true_stack, x_L, y_L, mask_left)
    right_null = _build_cluster_radial_null(true_stack, x_R, y_R, mask_right)

    return left_null, right_null

def get_blended_null_total(null_left, null_right, blend_width=0.05):
    """
    Smoothly blends null_left and null_right across x = 0 using a 
    sigmoid weight function to remove the boundary seam line.
    
    Parameters:
    - blend_width: Controls transition sharpness. Smaller = narrower transition zone.
                   0.05 radians (~3 pixels) creates a seamless boundary.
    """
    # Weight map for the right side (0 on left, 1 on right, 0.5 at x = 0)
    w_R = 1.0 / (1.0 + np.exp(-X_GRID / blend_width))
    w_L = 1.0 - w_R

    # Blended output map
    null_total_blended = w_L * null_left + w_R * null_right
    return null_total_blended

def _save_map(data, filename, title, cbar_label, grid_min=GRID_MIN, grid_max=GRID_MAX, contour_levels=None):
    plt.figure(figsize=(8, 7))
    vmin, vmax = np.percentile(data, [1, 99.5])
    im = plt.imshow(data, cmap='viridis', origin='lower',
                     extent=[grid_min, grid_max, grid_min, grid_max],
                     vmin=vmin, vmax=vmax)

    # Add contour lines
    if contour_levels is not None:
        cs = plt.contour(
            data, 
            levels=contour_levels, 
            colors='white',         # High contrast against viridis background
            linewidths=1.0, 
            linestyles='solid',
            extent=[grid_min, grid_max, grid_min, grid_max]
        )
        # Add small numerical labels on top of the lines
        plt.clabel(cs, inline=True, fontsize=8, fmt='%.4f', colors='white')

    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved to {filename}")

def _save_filament_map(data, filename, title, cbar_label, grid_min=GRID_MIN, grid_max=GRID_MAX, excl=(0.5, 0.3)):
    print("Plotting final subtracted mass map...")
    plt.figure(figsize=(8, 7))
    
    vmax = np.percentile(np.abs(data), 99.5)
    im = plt.imshow(data, cmap='viridis', origin='lower',
                    extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX], vmin=-vmax, vmax=vmax)
    plt.colorbar(im, pad=0.02).set_label(cbar_label, fontsize=12)

    ax = plt.gca()
    ax.add_patch(plt.Rectangle((-0.3, -0.15), 0.6, 0.3, lw=1.5, edgecolor='black', facecolor='none'))
    ax.add_patch(plt.Rectangle((-excl[0], -excl[1]), 2*excl[0], 2*excl[1], lw=1.2,
                               edgecolor='black', facecolor='none', linestyle='--'))
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

    # Mirror outer half to the right of x = -0.5
    outer_left_flipped = np.fliplr(true_stack[:, :left_idx])
    n_cols_L = outer_left_flipped.shape[1]
    left_null[:, left_idx + 1 : left_idx + 1 + n_cols_L] = outer_left_flipped

    # Reflect across the cluster center, but only out to x=0 -- beyond that
    # is the right cluster's / filament's territory, not the left cluster's
    
    # this part is w/ the previous method of mirroring;
    # i.e. cutting off the nulls at x=0
    # commenting it out to test a new version that allows
    # the null maps to extend past x=0 (making left and right
    # nulls separately)
    """
    inner_width_L = mid_idx - left_idx
    outer_slice_L = true_stack[:, left_idx + 1 - inner_width_L: left_idx + 1]
    left_null[:, left_idx + 1: left_idx + 1 + inner_width_L] = np.fliplr(outer_slice_L)
    """

    # RIGHT CLUSTER
    right_null[:, right_idx:] = true_stack[:, right_idx:]

    # Mirror outer half to the left of x = +0.5
    outer_right_flipped = np.fliplr(true_stack[:, right_idx + 1:])
    n_cols_R = outer_right_flipped.shape[1]
    right_null[:, right_idx - n_cols_R : right_idx] = outer_right_flipped

    """
    inner_width_R = right_idx - mid_idx
    outer_slice_R = true_stack[:, right_idx: right_idx + inner_width_R]
    right_null[:, right_idx - inner_width_R: right_idx] = np.fliplr(outer_slice_R)
    """

    return left_null, right_null

def nfw_kappa_2d(X, Y, center_x, center_y, kappa_0=0.08, r_s=0.25, q=1.0, phi_halo=0.0):
    """
    Computes projected NFW convergence kappa(x, y).
    - q: Axis ratio (b/a <= 1.0).
    - phi_halo: Halo position angle in radians relative to the x-axis.
    """
    dx = X - center_x
    dy = Y - center_y
    
    # Rotate by position angle
    cos_p, sin_p = np.cos(phi_halo), np.sin(phi_halo)
    dx_rot = dx * cos_p + dy * sin_p
    dy_rot = -dx * sin_p + dy * cos_p
    
    # Elliptical radius scale
    r = np.sqrt(dx_rot**2 + (dy_rot / q)**2) / r_s
    r = np.maximum(r, 1e-4)  # Avoid division by zero
    
    f = np.zeros_like(r)
    m_in = r < 1.0
    m_out = r > 1.0
    m_eq = np.isclose(r, 1.0)
    
    f[m_in] = (1.0 - (2.0 / np.sqrt(1.0 - r[m_in]**2)) * np.arctanh(np.sqrt((1.0 - r[m_in]) / (1.0 + r[m_in])))) / (r[m_in]**2 - 1.0)
    f[m_out] = (1.0 - (2.0 / np.sqrt(r[m_out]**2 - 1.0)) * np.arctan(np.sqrt((r[m_out] - 1.0) / (r[m_out] + 1.0)))) / (r[m_out]**2 - 1.0)
    f[m_eq] = 1.0 / 3.0
    
    return 2.0 * kappa_0 * f

def create_mock_healpix_map(df, base_nside=1024, inject_filament=False, f_amp=0.02, f_sigma=0.1,
                            q=1.0, align_halos=False, seed=42, include_left=True, include_right=True, 
                            subtract_mean=True):
    """
    Injects NFW halos (spherical or elliptical) and optional filaments onto a HEALPix map.
    """
    print(f"\nGenerating Mock HEALPix Map (NSIDE={base_nside}, Filament={inject_filament}, q={q}, Align={align_halos})...")
    npix = hp.nside2npix(base_nside)
    mock_map = np.zeros(npix)
    rng = np.random.default_rng(seed)

    num_pairs = len(df)
    for i, row in enumerate(df.itertuples()):
        if i % 1000 == 0 and i > 0:
            print(f"  Injected {i}/{num_pairs} mock pairs...")

        ra_L, dec_L = np.radians(row.ra_L), np.radians(row.dec_L)
        ra_R, dec_R = np.radians(row.ra_R), np.radians(row.dec_R)
        sep_rad = row.sep_rad

        # Midpoint calculation with explicit RA wrapping
        dra = (ra_R - ra_L + np.pi) % (2*np.pi) - np.pi
        alpha_mid = (ra_L + 0.5 * dra) % (2*np.pi)
        delta_mid = (dec_L + dec_R) / 2.0

        # Query radius expanded to 3.2 * sep_rad to safely cover full grid corners
        vec_mid = hp.ang2vec(np.pi/2 - delta_mid, alpha_mid)
        radius = 3.2 * sep_rad
        pix_indices = hp.query_disc(base_nside, vec_mid, radius)

        if len(pix_indices) == 0:
            continue

        theta_pix, phi_pix = hp.pix2ang(base_nside, pix_indices)
        dec_pix = np.pi/2 - theta_pix
        ra_pix = phi_pix

        # Proper wrapped RA coordinate subtraction for pixels
        dra_pix = (ra_pix - alpha_mid + np.pi) % (2*np.pi) - np.pi
        X_mid = -dra_pix * np.cos(delta_mid)
        Y_mid = dec_pix - delta_mid

        # Rotate to pair frame
        theta_pair = _theta_for_row(row)
        X_rot = X_mid * np.cos(-theta_pair) - Y_mid * np.sin(-theta_pair)
        Y_rot = X_mid * np.sin(-theta_pair) + Y_mid * np.cos(-theta_pair)

        X_norm = X_rot / sep_rad
        Y_norm = Y_rot / sep_rad

        # Halo orientation angles (aligned with x-axis vs random)
        if align_halos:
            phi_L, phi_R = 0.0, 0.0
        else:
            phi_L, phi_R = rng.uniform(0, np.pi, size=2)

        #k_L = nfw_kappa_2d(X_norm, Y_norm, center_x=-0.5, center_y=0.0, q=q, phi_halo=phi_L)
        #k_R = nfw_kappa_2d(X_norm, Y_norm, center_x=0.5, center_y=0.0, q=q, phi_halo=phi_R)
        #mock_map[pix_indices] += (k_L + k_R)

        kappa = np.zeros_like(X_norm)

        if include_left:
            kappa += nfw_kappa_2d(X_norm, Y_norm,center_x=-0.5,center_y=0.0,q=q,phi_halo=phi_L)

        if include_right:
            kappa += nfw_kappa_2d(X_norm, Y_norm,center_x=0.5,center_y=0.0,q=q,phi_halo=phi_R)

        mock_map[pix_indices] += kappa

        # Inject Gaussian filament bridge
        if inject_filament:
            f_mask = (X_norm >= -0.5) & (X_norm <= 0.5)
            filament = f_amp * np.exp(-(Y_norm**2) / (2 * f_sigma**2)) * f_mask
            mock_map[pix_indices] += filament

    # Subtract mean to match real pipeline pre-processing
    print(f"  Pre-subtraction mock_map mean: {np.mean(mock_map):.6e}")
    if subtract_mean:
        mock_map -= np.mean(mock_map)
    return mock_map


def _joint_design(n_nodes=40, n_nodes_m2=12, r_max=3.3,
                  excl=(0.5, 0.3), m2=True, ridge=1e-8):
    npix = GRID_RES * GRID_RES
    rows = np.arange(npix)

    def basis(xc, nodes, weight=None):
        R = np.hypot(X_GRID - xc, Y_GRID).ravel()
        idx = np.clip(np.searchsorted(nodes, R) - 1, 0, len(nodes) - 2)
        t = np.clip((R - nodes[idx]) / (nodes[idx + 1] - nodes[idx]), 0, 1)
        B = np.zeros((npix, len(nodes)))
        B[rows, idx] = 1 - t
        B[rows, idx + 1] = t
        if weight is not None:
            B *= weight[:, None]
        return B

    #r0 = np.linspace(0.0, r_max, n_nodes)
    #r2 = np.linspace(0.0, r_max, n_nodes_m2)
    
    r0 = np.unique(np.concatenate([np.linspace(0.0, 0.6, 31),     # spacing 0.02 in the core
                               np.linspace(0.6, r_max, 28)]))  # coarser in the wings
    r2 = np.unique(np.concatenate([np.linspace(0.0, 0.6, 7),
                               np.linspace(0.6, r_max, 8)]))

    blocks = [basis(-0.5, r0), basis(+0.5, r0)]
    if m2:
        for xc in (-0.5, +0.5):
            phi = np.arctan2(Y_GRID, X_GRID - xc).ravel()
            blocks.append(basis(xc, r2, weight=np.cos(2 * phi)))
    blocks.append(np.ones((npix, 1)))
    A = np.hstack(blocks)
    fit_mask = ~((np.abs(X_GRID) < excl[0]) & (np.abs(Y_GRID) < excl[1])).ravel()
    return A, fit_mask, ridge


def fit_two_halo_model(stack, **kw):
    A, fit_mask, ridge = _joint_design(**kw)
    Af, yf = A[fit_mask], stack.ravel()[fit_mask]
    coef = np.linalg.solve(Af.T @ Af + ridge * np.eye(A.shape[1]), Af.T @ yf)
    return (A @ coef).reshape(GRID_RES, GRID_RES)


def joint_box_weights(box=None, **kw):
    """W such that mean_box(stack - model(stack)) == sum(W * stack) exactly."""
    box = BOX_MASK if box is None else box
    A, fit_mask, ridge = _joint_design(**kw)
    Af = A[fit_mask]
    b = box.ravel().astype(float)
    b /= b.sum()
    G = Af.T @ Af + ridge * np.eye(A.shape[1])
    u = np.zeros(A.shape[0])
    u[fit_mask] = Af @ np.linalg.solve(G, A.T @ b)
    return (b - u).reshape(GRID_RES, GRID_RES)

def run_joint_analysis(dps_map, df):
    out = {}
    for name, kw in JOINT_CONFIGS.items():
        W = joint_box_weights(**kw)
        stack, _, stats = stack_true_filament(dps_map, df, weights=W)
        model = fit_two_halo_model(stack, **kw)
        direct = measure_filament_box(stack - model)[0]
        assert np.isclose(direct, stats.mean(), rtol=1e-6, atol=1e-10), (direct, stats.mean())
        #profile = extract_transverse_profile(per_pair_raw, model)
        out[name] = dict(stack=stack, model=model, filament=stack - model,
                         boot=compute_bootstrap_stats(stats), stats=stats)
        b = out[name]['boot']
        print(f"{name:16s} box = {direct:+.4e}  sigma = {b['sigma']:.2e}  S/N = {b['snr']:.2f}")
    return out


def shuffle_pairs(df, seed):
    """Move each pair (same separation and orientation) to another pair's L position."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    new = df.copy()
    ra_L, dec_L = np.radians(df.ra_L.values), np.radians(df.dec_L.values)
    ra_R, dec_R = np.radians(df.ra_R.values), np.radians(df.dec_R.values)
    dra = (ra_R - ra_L + np.pi) % (2*np.pi) - np.pi
    ddec = dec_R - dec_L
    nra_L, ndec_L = ra_L[perm], dec_L[perm]
    nra_R = nra_L + dra * np.cos(dec_L) / np.cos(ndec_L)
    ndec_R = ndec_L + ddec
    new['ra_L'], new['dec_L'] = np.degrees(nra_L) % 360, np.degrees(ndec_L)
    new['ra_R'], new['dec_R'] = np.degrees(nra_R) % 360, np.degrees(ndec_R)
    return new

def run_filament_recovery_benchmark(df, nside=1024, f_amp=0.05, f_sigma=0.1,
                                    num_rotations=1000, q=1.0, align_halos=False):
    print("\n" + "=" * 70)
    print(f"FILAMENT RECOVERY BENCHMARK (q={q}, align={align_halos}, rot={num_rotations})")
    print("=" * 70)

    # Raw maps, one common mean subtraction -> stacking is exactly linear
    hal = create_mock_healpix_map(df, base_nside=nside, q=q, align_halos=align_halos,
                                  subtract_mean=False)
    fil = create_mock_healpix_map(df, base_nside=nside, inject_filament=True,
                                  f_amp=f_amp, f_sigma=f_sigma,
                                  include_left=False, include_right=False,
                                  subtract_mean=False)
    m = np.mean(hal)
    mock_hal = hal - m
    mock_fil = hal + fil - m

    true_hal, _ = stack_true_filament(mock_hal, df, smooth=False)
    true_fil, _ = stack_true_filament(mock_fil, df, smooth=False)
    true_inj, _ = stack_true_filament(fil, df, smooth=False)   # exact injected signal

    inj = measure_filament_box(true_inj)[0]
    lin_err = measure_filament_box(true_fil - true_hal - true_inj)[0]
    print(f"Injected box kappa: {inj:+.6e}   (linearity check: {lin_err:+.2e})")

    def m4(mock, ts):
        L, _, _ = stack_cluster_null(mock, df, 'L', num_rotations=num_rotations,
                                     exclude_companion=False, smooth_mode='none')
        R, _, _ = stack_cluster_null(mock, df, 'R', num_rotations=num_rotations,
                                     exclude_companion=False, smooth_mode='none')
        return get_2halo_linear_null(L, R)

    def m6(mock, ts):
        return get_2halo_linear_null(*get_radial_profile_nulls(ts))

    def joint(mock, ts):
        return fit_two_halo_model(ts, m2=False)

    def jointm2(mock, ts):
        return fit_two_halo_model(ts)

    methods = {"Joint fit": joint, "Joint fit + m2": jointm2}

    rows = []
    for name, fn in methods.items():
        bias = measure_filament_box(true_hal - fn(mock_hal, true_hal))[0]   # no filament
        rec  = measure_filament_box(true_fil - fn(mock_fil, true_fil))[0]   # with filament
        rows.append({
            "Method": name,
            "Bias (no fil)": bias,
            "Recovered": rec,
            "R_raw = rec/inj": rec / inj,
            "R_resp = (rec-bias)/inj": (rec - bias) / inj,
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out

def run_exclusion_radius_scan(df_one, nside=1024):
    """
    Tests how the cross-halo contamination and recovered two-halo
    signal depend on the companion exclusion radius.

    exclusion radius = exclusion_frac * pair separation.
    """

    print("\n" + "=" * 75)
    print("COMPANION EXCLUSION-RADIUS SCAN")
    print("=" * 75)

    # Create the same spherical two-halo mock
    mock_LR = create_mock_healpix_map(df_one,base_nside=nside,inject_filament=False,
                                      q=1.0,include_left=True,include_right=True)
    true_stack, _ = stack_true_filament(mock_LR,df_one,smooth=False)

    # Scan exclusion radius
    fractions = [
        0.0,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00
    ]

    results = []

    for frac in fractions:
        print(f"\nTesting exclusion fraction = {frac:.2f}")
        null_L, _, _ = stack_cluster_null(mock_LR,df_one,target_side='L',num_rotations=1000,
                                          exclude_companion=(frac > 0),exclusion_frac=frac,smooth_mode='none')
        null_R, _, _ = stack_cluster_null(mock_LR,df_one,target_side='R',num_rotations=1000,
                                          exclude_companion=(frac > 0),exclusion_frac=frac,smooth_mode='none')

        # Raw two-halo null
        null_sum = null_L + null_R
        residual = measure_filament_box(true_stack - null_sum)[0]

        # Cross contamination relative to isolated-halo nulls
        mock_L = create_mock_healpix_map(df_one,base_nside=nside,inject_filament=False,
                                         q=1.0,include_left=True,include_right=False)
        mock_R = create_mock_healpix_map(df_one,base_nside=nside,inject_filament=False,
                                         q=1.0,include_left=False,include_right=True)

        iso_L, _, _ = stack_cluster_null(mock_L,df_one,target_side='L',num_rotations=1000,
                                         exclude_companion=False,smooth_mode='none')
        iso_R, _, _ = stack_cluster_null(mock_R,df_one,target_side='R',num_rotations=1000,
                                         exclude_companion=False,smooth_mode='none')

        cross_L = measure_filament_box(null_L - iso_L)[0]
        cross_R = measure_filament_box(null_R - iso_R)[0]
        cross_total = cross_L + cross_R

        print(f"  Box residual:          {residual:+.6e}")
        print(f"  L contamination:       {cross_L:+.6e}")
        print(f"  R contamination:       {cross_R:+.6e}")
        print(f"  Total contamination:   {cross_total:+.6e}")

        results.append({
            'exclusion_frac': frac,
            'residual': residual,
            'cross_L': cross_L,
            'cross_R': cross_R,
            'cross_total': cross_total
        })

    results_df = pd.DataFrame(results)

    print("\n" + "-" * 75)
    print(results_df.to_string(index=False))
    print("=" * 75)

    return results_df

def run_single_pair_null_diagnostic(df_one, nside=1024):
    """
    Controlled single-pair diagnostic.

    Tests:
      1. Isolated spherical left halo
      2. Isolated spherical right halo
      3. Combined spherical two-halo system
      4. Masked vs unmasked null
      5. Smoothing-order dependence
      6. Rotation-number dependence
      7. Method-4 background correction

    This is a diagnostic.
    """

    print("\n" + "=" * 75)
    print("SINGLE-PAIR NULL DIAGNOSTIC")
    print("=" * 75)

    # 1. CREATE RAW MOCKS
    # Do NOT subtract the mean separately from L and R.
    
    mock_L_raw = create_mock_healpix_map(df_one,base_nside=nside,inject_filament=False,
                                         q=1.0,include_left=True,include_right=False,subtract_mean=False)
    mock_R_raw = create_mock_healpix_map(df_one,base_nside=nside,inject_filament=False,
                                         q=1.0,include_left=False,include_right=True,subtract_mean=False)
    mock_LR_raw = create_mock_healpix_map(df_one,base_nside=nside,inject_filament=False,
                                          q=1.0,include_left=True,include_right=True,subtract_mean=False)

    # Sanity check: LR should equal L + R
    reconstruction_error = np.max(np.abs(mock_LR_raw - (mock_L_raw + mock_R_raw)))
    print(f"\nRaw mock consistency check: "f"max|LR - (L+R)| = {reconstruction_error:.3e}")

    # 2. APPLY ONE COMMON SKY-MEAN SUBTRACTION

    common_mean = np.mean(mock_LR_raw)

    mock_L = mock_L_raw - common_mean
    mock_R = mock_R_raw - common_mean
    mock_LR = mock_LR_raw - common_mean

    print(f"Common mock sky mean = {common_mean:+.6e}")

    # 3. ANALYTIC NFW EXPECTATION

    analytic_L = nfw_kappa_2d(X_GRID, Y_GRID,center_x=-0.5,center_y=0.0,q=1.0)
    analytic_R = nfw_kappa_2d(X_GRID, Y_GRID,center_x=+0.5,center_y=0.0,q=1.0)

    analytic_total = analytic_L + analytic_R

    # Since the mock sky has been mean-subtracted,
    # include the SAME constant offset in the analytic expectation.
    analytic_total_zero_mean = analytic_total - common_mean

    # 4. TRUE STACKS

    true_L, _ = stack_true_filament(mock_L,df_one,smooth=False)
    true_R, _ = stack_true_filament(mock_R,df_one,smooth=False)
    true_LR, _ = stack_true_filament(mock_LR,df_one,smooth=False)

    print("\nTRUE STACK VS EXPECTED ANALYTIC MAP")
    print(f"  L true - analytic L:"f" {measure_filament_box(true_L - (analytic_L - common_mean))[0]:+.6e}")
    print(f"  R true - analytic R:"f" {measure_filament_box(true_R - (analytic_R - common_mean))[0]:+.6e}")
    print(f"  LR true - analytic total:"f" {measure_filament_box(true_LR - analytic_total_zero_mean)[0]:+.6e}")

    # 5. ISOLATED ONE-HALO NULL TESTS
    #
    # NO companion masking here.
    #       Does rotating a spherical halo preserve it?

    print("\n" + "-" * 75)
    print("TEST 1: ISOLATED SPHERICAL HALO")
    print("-" * 75)

    for side, mock_single, true_single in [('L', mock_L, true_L),('R', mock_R, true_R)]:
        null_single, _, _ = stack_cluster_null(mock_single,df_one,target_side=side,num_rotations=1000,
                                               exclude_companion=False,smooth_mode='none')
        isolated_bias = measure_filament_box(true_single - null_single)[0]
        print(f"  {side}-halo true - null = "f"{isolated_bias:+.6e}")

    # 5.1. CROSS-HALO CONTAMINATION

    print("\n" + "-" * 75)
    print("TEST 1B: CROSS-HALO CONTAMINATION")
    print("-" * 75)

    # Isolated nulls
    null_L_iso, _, _ = stack_cluster_null(mock_L,df_one,target_side='L',num_rotations=1000,
                                          exclude_companion=False,smooth_mode='none')
    null_R_iso, _, _ = stack_cluster_null(mock_R,df_one,target_side='R',num_rotations=1000,
                                          exclude_companion=False,smooth_mode='none')

    iso_null_sum = null_L_iso + null_R_iso
    print("True - isolated null sum:",measure_filament_box(true_LR - iso_null_sum)[0])

    # Combined nulls
    null_L_comb, _, _ = stack_cluster_null(mock_LR,df_one,target_side='L',num_rotations=1000,
                                           exclude_companion=False,smooth_mode='none')
    null_R_comb, _, _ = stack_cluster_null(mock_LR,df_one,target_side='R',num_rotations=1000,
                                           exclude_companion=False,smooth_mode='none')

    # Cross-contamination maps
    cross_L = null_L_comb - null_L_iso
    cross_R = null_R_comb - null_R_iso

    cross_total = cross_L + cross_R

    print(f"L-null cross-contamination: "f"{measure_filament_box(cross_L)[0]:+.6e}")
    print(f"R-null cross-contamination: "f"{measure_filament_box(cross_R)[0]:+.6e}")
    print(f"Total cross-contamination: "f"{measure_filament_box(cross_total)[0]:+.6e}")

    _save_map(cross_L, 
              os.path.join(OUTPUT_DIR, f'cross_contamination_L.png'),
              "Cross-Halo Contamination in L Null",r"$\Delta\kappa$")

    _save_map(cross_R,
              os.path.join(OUTPUT_DIR, f'cross_contamination_R.png'),
              "Cross-Halo Contamination in R Null",r"$\Delta\kappa$")

    _save_map(cross_total,
              os.path.join(OUTPUT_DIR, f'cross_contamination_total.png'),
              "Total Cross-Halo Contamination",r"$\Delta\kappa$")

    # 6. COMBINED TWO-HALO TEST
    # First WITHOUT masking.
    # Then WITH masking.
    # No smoothing initially.

    print("\n" + "-" * 75)
    print("TEST 2: TWO HALOS — MASKED VS UNMASKED")
    print("-" * 75)

    combined_results = {}

    for exclude in [False, True]:
        label = "masked" if exclude else "unmasked"
        print(f"\n  --- {label.upper()} ---")
        null_L, _, _ = stack_cluster_null(mock_LR,df_one,target_side='L',num_rotations=1000,
                                          exclude_companion=exclude,smooth_mode='none')
        null_R, _, _ = stack_cluster_null(mock_LR,df_one,target_side='R',num_rotations=1000,
                                          exclude_companion=exclude,smooth_mode='none')
        null_sum = null_L + null_R

        # Raw sum
        raw_residual = measure_filament_box(true_LR - null_sum)[0]

        # Method 4
        null_m4 = get_2halo_linear_null(null_L,null_R)
        m4_residual = measure_filament_box(true_LR - null_m4)[0]

        # Individual diagnostics
        true_box = measure_filament_box(true_LR)[0]
        null_L_box = measure_filament_box(null_L)[0]
        null_R_box = measure_filament_box(null_R)[0]
        null_sum_box = measure_filament_box(null_sum)[0]
        null_m4_box = measure_filament_box(null_m4)[0]

        print(f"  True LR box:          {true_box:+.6e}")
        print(f"  Null L box:           {null_L_box:+.6e}")
        print(f"  Null R box:           {null_R_box:+.6e}")
        print(f"  Null L+R box:         {null_sum_box:+.6e}")
        print(f"  Method 4 null box:    {null_m4_box:+.6e}")
        print(f"  True - raw null:      {raw_residual:+.6e}")
        print(f"  True - Method 4:      {m4_residual:+.6e}")

        combined_results[label] = {
            'true': true_box,
            'null_L': null_L_box,
            'null_R': null_R_box,
            'null_sum': null_sum_box,
            'm4_null': null_m4_box,
            'raw_residual': raw_residual,
            'm4_residual': m4_residual
            }

    # 7. SMOOTHING ORDER TEST
    #
    # This is ONLY done on the combined two-halo map.

    print("\n" + "-" * 75)
    print("TEST 3: SMOOTHING ORDER")
    print("-" * 75)

    for mode in ['none', 'after_mask', 'before_mask']:
        # When mode='none', use no smoothing.
        # Otherwise use the requested ordering.
        null_L, _, _ = stack_cluster_null(mock_LR,df_one,target_side='L',num_rotations=1000,
                                          exclude_companion=True,smooth_mode=mode)
        null_R, _, _ = stack_cluster_null(mock_LR,df_one,target_side='R',num_rotations=1000,
                                          exclude_companion=True,smooth_mode=mode)
        null_m4 = get_2halo_linear_null(null_L,null_R)
        bias = measure_filament_box(true_LR - null_m4)[0]
        print(f"  {mode:15s}: "f"{bias:+.6e}")

    # 8. ROTATION-COUNT TEST

    print("\n" + "-" * 75)
    print("TEST 4: NUMBER OF RANDOM ROTATIONS")
    print("-" * 75)

    for nrot in [10, 100, 1000]:
        null_L, _, _ = stack_cluster_null(mock_LR,df_one,target_side='L',num_rotations=nrot,
                                          exclude_companion=True,smooth_mode='none')
        null_R, _, _ = stack_cluster_null(mock_LR,df_one,target_side='R',num_rotations=nrot,
                                          exclude_companion=True,smooth_mode='none')
        null_m4 = get_2halo_linear_null(null_L,null_R)
        bias = measure_filament_box(true_LR - null_m4)[0]

        print(f"  {nrot:4d} rotations: "f"{bias:+.6e}")

    print("\n" + "=" * 75)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 75)

    return combined_results

def run_pipeline_benchmark(df, nside=1024):
    """
    Executes Test A (Bias & False Positives) and Test B (Filament Recovery).
    """
    print("\n" + "="*70)
    print("PIPELINE END-TO-END MOCK BENCHMARK -- SMOOTHING DISABLEDT")
    print("="*70)

    # TEST A1: Spherical Halos, No Filament (Zero-Bias Check)
    print("\n[Test A1: Spherical Halos (q=1.0), No Filament -> Bias Check]")
    mock_null_sph = create_mock_healpix_map(df, base_nside=nside, inject_filament=False, q=1.0)

    true_sph, _ = stack_true_filament(mock_null_sph, df, smooth=False)
    null_L_m4, _, _ = stack_cluster_null(mock_null_sph, df, 'L', exclude_companion=True, smooth_mode='none')
    null_R_m4, _, _ = stack_cluster_null(mock_null_sph, df, 'R', exclude_companion=True, smooth_mode='none')
    null_m4_sph = get_2halo_linear_null(null_L_m4, null_R_m4)

    null_L_rad, null_R_rad = get_radial_profile_nulls(true_sph)
    null_m6_sph = get_2halo_linear_null(null_L_rad, null_R_rad)

    bias_m4_sph = measure_filament_box(true_sph - null_m4_sph)[0]
    bias_m6_sph = measure_filament_box(true_sph - null_m6_sph)[0]
    print(f"  Method 4 Bias (Spherical): {bias_m4_sph:+.6e}")
    print(f"  Method 6 Bias (Spherical): {bias_m6_sph:+.6e}")

    # TEST A2: Pair-Aligned Elliptical Halos (False-Positive Check)
    print("\n[Test A2: Pair-Aligned Elliptical Halos (q=0.7), No Filament -> False-Positive Check]")
    mock_null_ell = create_mock_healpix_map(df, base_nside=nside, inject_filament=False, q=0.7, align_halos=True)

    true_ell, _ = stack_true_filament(mock_null_ell, df, smooth=False)
    null_L_m4_e, _, _ = stack_cluster_null(mock_null_ell, df, 'L', exclude_companion=True, smooth_mode='none')
    null_R_m4_e, _, _ = stack_cluster_null(mock_null_ell, df, 'R', exclude_companion=True, smooth_mode='none')
    null_m4_ell = get_2halo_linear_null(null_L_m4_e, null_R_m4_e)

    null_L_rad_e, null_R_rad_e = get_radial_profile_nulls(true_ell)
    null_m6_ell = get_2halo_linear_null(null_L_rad_e, null_R_rad_e)

    bias_m4_ell = measure_filament_box(true_ell - null_m4_ell)[0]
    bias_m6_ell = measure_filament_box(true_ell - null_m6_ell)[0]
    print(f"  Method 4 Bias (Aligned Elliptical): {bias_m4_ell:+.6e}")
    print(f"  Method 6 Bias (Aligned Elliptical): {bias_m6_ell:+.6e}")

    # TEST A3: Random Elliptical Halos 
    print("\n[Test A3: Random Elliptical Halos (q=0.7), No Filament]")
    mock_null_ell2 = create_mock_healpix_map(df, base_nside=nside, inject_filament=False, q=0.7, align_halos=False)
    
    true_ell2, _ = stack_true_filament(mock_null_ell2, df, smooth=False)
    null_L_m4_e2, _, _ = stack_cluster_null(mock_null_ell2, df, 'L', exclude_companion=True, smooth_mode='none')
    null_R_m4_e2, _, _ = stack_cluster_null(mock_null_ell2, df, 'R', exclude_companion=True, smooth_mode='none')
    null_m4_ell2 = get_2halo_linear_null(null_L_m4_e2, null_R_m4_e2)
    
    null_L_rad_e2, null_R_rad_e2 = get_radial_profile_nulls(true_ell2)
    null_m6_ell2 = get_2halo_linear_null(null_L_rad_e2, null_R_rad_e2)

    bias_m4_ell2 = measure_filament_box(true_ell2 - null_m4_ell2)[0]
    bias_m6_ell2 = measure_filament_box(true_ell2 - null_m6_ell2)[0]
    print(f"  Method 4 Bias (Random Elliptical): {bias_m4_ell2:+.6e}")
    print(f"  Method 6 Bias (Random Elliptical): {bias_m6_ell2:+.6e}")

    # TEST B: Filament Recovery Ratio (Signal Recovery Check)
    F_AMP, F_SIGMA = 0.05, 0.1
    print(f"\n[Test B: Filament Recovery Ratio (Injected Amp = {F_AMP})]")
    mock_fil = create_mock_healpix_map(df, base_nside=nside, inject_filament=True, f_amp=F_AMP, f_sigma=F_SIGMA, q=1.0)

    true_fil, _ = stack_true_filament(mock_fil, df, smooth=False)
    null_L_m4_f, _, _ = stack_cluster_null(mock_fil, df, 'L', exclude_companion=True, smooth_mode='none')
    null_R_m4_f, _, _ = stack_cluster_null(mock_fil, df, 'R', exclude_companion=True, smooth_mode='none')
    null_m4_fil = get_2halo_linear_null(null_L_m4_f, null_R_m4_f)

    null_L_rad_f, null_R_rad_f = get_radial_profile_nulls(true_fil)
    null_m6_fil = get_2halo_linear_null(null_L_rad_f, null_R_rad_f)

    # Injected signal measured through the true stack pipeline (True_Filament - True_Null)
    injected_true_stack = true_fil - true_sph
    injected_box_kappa = measure_filament_box(injected_true_stack)[0]

    rec_m4 = measure_filament_box(true_fil - null_m4_fil)[0]
    rec_m6 = measure_filament_box(true_fil - null_m6_fil)[0]

    R4 = rec_m4 / injected_box_kappa
    R6 = rec_m6 / injected_box_kappa

    print(f"  Pipeline Injected Box Kappa: {injected_box_kappa:+.6e}")
    print(f"  Method 4 Recovery Ratio (R): {R4:.4f}")
    print(f"  Method 6 Recovery Ratio (R): {R6:.4f}")
    print("="*70 + "\n")


def measure_filament_box(filament_map, x_range=(-0.3, 0.3), y_range=(-0.15, 0.15)):
    """
    Calculates the mean and integrated kappa signal inside the designated filament box.
    """
    box_mask = (
        (X_GRID >= x_range[0]) & (X_GRID <= x_range[1]) &
        (Y_GRID >= y_range[0]) & (Y_GRID <= y_range[1])
    )
    
    mean_kappa = np.mean(filament_map[box_mask])
    # Area-integrated convergence: sum(kappa * dA)
    integrated_kappa = np.sum(filament_map[box_mask]) * (PIXEL_WIDTH ** 2)
    return mean_kappa, integrated_kappa

def run_systematics_check(true_stack, true_box_kappas, dps_map, df):
    print("RUNNING NULL METHOD SYSTEMATICS CHECK")

    # 1. Generate base null building blocks (returning both average stack and per-pair arrays)
    null_L_unm, box_L_unm, bg_L_unm = stack_cluster_null(dps_map, df, target_side='L', exclude_companion=False)
    null_R_unm, box_R_unm, bg_R_unm = stack_cluster_null(dps_map, df, target_side='R', exclude_companion=False)

    null_L_msk, box_L_msk, bg_L_msk = stack_cluster_null(dps_map, df, target_side='L', exclude_companion=True)
    null_R_msk, box_R_msk, bg_R_msk = stack_cluster_null(dps_map, df, target_side='R', exclude_companion=True)

    null_L_rad, null_R_rad = get_radial_profile_nulls(true_stack)
    null_L_mirr, null_R_mirr = get_mirrored_nulls(true_stack)

    # 2. Linear combination of per-pair box values
    bg_mean_unm = 0.5 * (bg_L_unm + bg_R_unm)
    bg_mean_msk = 0.5 * (bg_L_msk + bg_R_msk)

    per_pair_null_kappas = {
        "1. Naive Sum (Unmasked)": box_L_unm + box_R_unm,
        "2. Naive Sum (Masked)": box_L_msk + box_R_msk,
        "3. Linear 2-Halo (Unmasked)": box_L_unm + box_R_unm - bg_mean_unm,
        "4. Linear 2-Halo (Masked)": box_L_msk + box_R_msk - bg_mean_msk,
        "5. Blended Half-Plane": None,
        "6. 1D Radial Profile Reconstruction": None,
        "7. Mirrored Null": None,
    }

    # Full average maps for box measurement
    null_maps = {
        "1. Naive Sum (Unmasked)": null_L_unm + null_R_unm,
        "2. Naive Sum (Masked)": null_L_msk + null_R_msk,
        "3. Linear 2-Halo (Unmasked)": get_2halo_linear_null(null_L_unm, null_R_unm),
        "4. Linear 2-Halo (Masked)": get_2halo_linear_null(null_L_msk, null_R_msk),
        "5. Blended Half-Plane": get_blended_null_total(null_L_msk, null_R_msk),
        "6. 1D Radial Profile Reconstruction": get_2halo_linear_null(null_L_rad, null_R_rad),
        "7. Mirrored Null": null_L_mirr + null_R_mirr,
    }

    results = []
    for name, null_map in null_maps.items():
        filament_map = true_stack - null_map
        mean_k, int_k = measure_filament_box(filament_map)

        per_pair_null = per_pair_null_kappas[name]
        # Calculate Bootstrap Stats if per-pair data exists
        if per_pair_null is not None:
            # Subtracted per-pair residual box kappa values
            per_pair_residuals = true_box_kappas - per_pair_null
            boot_stats = compute_bootstrap_stats(per_pair_residuals)
            sigma_k = boot_stats['sigma']
            snr = boot_stats['snr']
        else:
            sigma_k = np.nan
            snr = np.nan

        results.append({
            'Method': name,
            'Mean Box Kappa': mean_k,
            'Sigma Kappa': sigma_k,
            'S/N Ratio': snr,
            'Integrated Area Kappa': int_k,
        })

    df_results = pd.DataFrame(results)

    baseline_val = df_results.loc[df_results['Method'] == "4. Linear 2-Halo (Masked)", 'Mean Box Kappa'].values[0]
    df_results['Shift vs Linear 2-Halo (%)'] = ((df_results['Mean Box Kappa'] - baseline_val) / np.abs(baseline_val)) * 100.0

    print("\nSYSTEMATICS SUMMARY TABLE (WITH BOOTSTRAP ERRORS):")
    print(df_results[['Method', 'Mean Box Kappa', 'Sigma Kappa', 'S/N Ratio', 'Shift vs Linear 2-Halo (%)']].to_string(index=False))

    return df_results

def _shuffle_worker(seed):
    stack, _ = stack_true_filament(_G['dps'], shuffle_pairs(_G['df'], seed))
    # estimator is linear: box value = sum(W * stack), so one stack gives both fits
    return [float(np.sum(W * stack)) for W in _G['W']]

def run_shuffle_test(n_shuffles=N_SHUFFLES, n_proc=N_PROC):
    df = pd.read_csv(PAIRS_FILE)
    dps = hp.read_map(DPS_MAP_FILE)
    dps = dps - np.mean(dps)
    names = list(JOINT_CONFIGS)
    Ws = [joint_box_weights(**JOINT_CONFIGS[n]) for n in names]
    _G.update(dps=dps, df=df, W=Ws)            # set BEFORE creating the Pool (fork)

    real_stack, _ = stack_true_filament(dps, df)
    real = [float(np.sum(W * real_stack)) for W in Ws]

    with Pool(n_proc) as p:
        res = np.array(p.map(_shuffle_worker, range(n_shuffles)))
    np.save(os.path.join(OUTPUT_DIR, f'shuffle_results_tomo{TOMO_BIN}.npy'), res)

    for j, n in enumerate(names):
        s = res[:, j]
        err = s.std(ddof=1) / np.sqrt(len(s))
        p_val = (1 + np.sum(np.abs(s) >= abs(real[j]))) / (len(s) + 1)
        print(f"{n:16s} real = {real[j]:+.3e} | shuffles: mean = {s.mean():+.3e} "
              f"+/- {err:.1e} ({s.mean()/err:+.2f} sigma), scatter = {s.std(ddof=1):.3e} "
              f"| empirical p = {p_val:.3f}")

def run_width_test(sigmas=(0.05, 0.1, 0.2, 0.3, 0.4),
                   excls=((0.5, 0.3), (0.5, 0.4), (0.5, 0.5),(0.5, 1.0)), amp=0.05):
    """Filament response vs. width and exclusion region.
    The estimator is linear, so the response can be measured on a filament-only
    map on the pair-normalized grid: no halos or HEALPix mock needed."""
    f_mask = np.abs(X_GRID) <= 0.5
    rows = []
    for s in sigmas:
        fil = _smooth(amp * np.exp(-Y_GRID**2 / (2 * s**2)) * f_mask)
        inj = measure_filament_box(fil)[0]
        for ex in excls:
            for name, kw in JOINT_CONFIGS.items():
                rec = measure_filament_box(fil - fit_two_halo_model(fil, excl=ex, **kw))[0]
                rows.append(dict(f_sigma=s, excl=str(ex), method=name, R_resp=rec / inj))
    out = pd.DataFrame(rows)
    print(out.pivot_table(index=['f_sigma', 'excl'], columns='method', values='R_resp').round(3))
    return out

def run_excl_robustness_test(dps_map, df, excls=((0.5,0.3),(0.5,0.4),(0.5,0.5),(0.5, 1.0))):
    stack, _ = stack_true_filament(dps_map, df)
    for ex in excls:
        for name, kw in JOINT_CONFIGS.items():
            v = measure_filament_box(stack - fit_two_halo_model(stack, excl=ex, **kw))[0]
            print(f"excl={ex}  {name:16s} box = {v:+.3e}")

def run_halves_test(dps_map, df, m2=True):
    """Box signal in the left (x<0) and right (x>0) halves of the bridge box."""
    kw = dict(m2=m2)
    boxes = {"left":  BOX_MASK & (X_GRID < 0),
             "right": BOX_MASK & (X_GRID > 0)}
    for label, box in boxes.items():
        W = joint_box_weights(box=box, **kw)
        stack, _, stats = stack_true_filament(dps_map, df, weights=W)
        model = fit_two_halo_model(stack, **kw)
        direct = np.mean((stack - model)[box])
        assert np.isclose(direct, stats.mean(), rtol=1e-6, atol=1e-10), (direct, stats.mean())
        b = compute_bootstrap_stats(stats)
        print(f"{label:5s} half: box = {direct:+.3e}  sigma = {b['sigma']:.2e}  S/N = {b['snr']:.2f}")

def run_xprofile(dps_map, df, m2=True, edges=np.linspace(-0.5, 0.5, 6)):
    kw = dict(m2=m2)
    for lo, hi in zip(edges[:-1], edges[1:]):
        box = (X_GRID >= lo) & (X_GRID < hi) & (np.abs(Y_GRID) <= 0.15)
        W = joint_box_weights(box=box, **kw)
        stack, _, st = stack_true_filament(dps_map, df, weights=W)
        b = compute_bootstrap_stats(st)
        print(f"x in [{lo:+.1f},{hi:+.1f}): {st.mean():+.2e} +/- {b['sigma']:.1e}")

def run_profile(dps_map, df, axis='x', m2=True, excl=(0.5, 0.3), edges=None):
    """Signal in strips along x (|y|<=0.15) or along y (|x|<=0.3). Use excl=(0.5, 1.0)
    for axis='y', otherwise pixels beyond |y|=0.3 are fitted and read ~0."""
    kw = dict(m2=m2, excl=excl)
    if edges is None:
        edges = np.linspace(-0.5, 0.5, 6) if axis == 'x' else np.linspace(-0.6, 0.6, 9)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if axis == 'x':
            box = (X_GRID >= lo) & (X_GRID < hi) & (np.abs(Y_GRID) <= 0.15)
        else:
            box = (Y_GRID >= lo) & (Y_GRID < hi) & (np.abs(X_GRID) <= 0.3)
        W = joint_box_weights(box=box, **kw)
        _, _, st = stack_true_filament(dps_map, df, weights=W)
        b = compute_bootstrap_stats(st)
        print(f"{axis} in [{lo:+.2f},{hi:+.2f}): {st.mean():+.2e} +/- {b['sigma']:.1e}")

# this does NOT work dont use it itll kill the process IDK WHYYYYY
"""
def extract_transverse_profile(per_pair_raw, model, x_range=(-0.3, 0.3), n_bootstraps=1000, seed=42):
    #Fast 1D transverse profile & bootstrap error calculation.
    x_mask = (X_GRID[0, :] >= x_range[0]) & (X_GRID[0, :] <= x_range[1])
    n_pairs = len(per_pair_raw)
    
    # Subtract model profile from raw pair profiles
    model_profile = np.mean(model[:, x_mask], axis=1)  # shape: (GRID_RES,)
    per_pair_profiles = per_pair_raw - model_profile   # shape: (n_pairs, GRID_RES)

    # Vectorized bootstrap resampling
    rng = np.random.default_rng(seed)
    boot_idx = rng.choice(n_pairs, size=(n_bootstraps, n_pairs), replace=True)
    boot_profiles = np.mean(per_pair_profiles[boot_idx], axis=1)

    return {
        'y': Y_LIN,
        'mean': np.mean(per_pair_profiles, axis=0),
        'sigma': np.std(boot_profiles, axis=0, ddof=1),
        'ci_lower': np.percentile(boot_profiles, 15.85, axis=0),
        'ci_upper': np.percentile(boot_profiles, 84.15, axis=0)
    }

def plot_transverse_profile(profile, filename, title):
    plt.figure(figsize=(7, 5))
    y = profile['y']
    plt.plot(y, profile['mean'], color='crimson', lw=2, label=r'Transverse $\kappa(y)$')
    plt.fill_between(y, profile['ci_lower'], profile['ci_upper'], color='crimson', alpha=0.25, label=r'1$\sigma$ Bootstrap')

    plt.axhline(0, color='gray', linestyle='--', lw=1)
    plt.axvline(0, color='gray', linestyle=':', lw=1)
    plt.axvline(-0.15, color='black', linestyle='--', lw=1, alpha=0.6, label='Box Y-bound')
    plt.axvline(0.15, color='black', linestyle='--', lw=1, alpha=0.6)

    plt.xlim(-1.0, 1.0)
    plt.xlabel('y', fontsize=12)
    plt.ylabel(r'Convergence $\kappa(y)$', fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
"""

def run_halves():
    df = pd.read_csv(PAIRS_FILE)
    dps = hp.read_map(DPS_MAP_FILE)
    dps = dps - np.mean(dps)
    run_halves_test(dps, df, m2=True)

def run_excl_robustness():
    df = pd.read_csv(PAIRS_FILE)
    dps = hp.read_map(DPS_MAP_FILE)
    dps = dps - np.mean(dps)
    run_excl_robustness_test(dps, df)


def main_real():
    t0 = time.perf_counter()
    df = pd.read_csv(PAIRS_FILE)
    dps_map = hp.read_map(DPS_MAP_FILE)
    dps_map = dps_map - np.mean(dps_map)

    res = run_joint_analysis(dps_map, df)

    # maps: one set per fit, separate filenames
    for key, tag in [("Joint fit", "plain"), ("Joint fit + m2", "m2")]:
        r = res[key]
        base = f'tomo{TOMO_BIN}{FILE_SUFFIX}_{tag}'
        _save_map(r['stack'], os.path.join(OUTPUT_DIR, f'dps_true_stack_{base}.png'),
                  f'Stacked DPS (Tomo {TOMO_BIN})', r'DPS $\kappa$')
        _save_map(r['model'], os.path.join(OUTPUT_DIR, f'dps_null_total_{base}.png'),
                  f'Joint 2-halo model, {tag} (Tomo {TOMO_BIN})', r'DPS $\kappa$')
        _save_filament_map(r['filament'], os.path.join(OUTPUT_DIR, f'dps_filament_map_{base}.png'),
                           f'Subtracted map, {tag} (Tomo {TOMO_BIN})', r'DPS $\kappa$')
        np.savez(os.path.join(OUTPUT_DIR, f'maps_{base}.npz'),
                 stack=r['stack'], model=r['model'], filament=r['filament'])

    # one row per (bin, fit), appended so bins can be compared later
    rows = []
    for key, r in res.items():
        rows.append(dict(tomo=TOMO_BIN, method=key,
                         box=r['stats'].mean(),          # equals the direct box value (asserted)
                         sigma_boot=r['boot']['sigma'],
                         snr=r['stats'].mean() / r['boot']['sigma'],
                         ci68_lo=r['boot']['ci_68'][0], ci68_hi=r['boot']['ci_68'][1]))
    out = pd.DataFrame(rows)
    csv = os.path.join(OUTPUT_DIR, 'bin_summary.csv')
    out.to_csv(csv, mode='a', header=not os.path.exists(csv), index=False)

    d = res["Joint fit + m2"]['stats'].mean() - res["Joint fit"]['stats'].mean()
    print(f"m2 - plain joint (alignment systematic): {d:+.3e}")
    print(f"Total time: {(time.perf_counter()-t0)/60:.1f} min")

def main():
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "real"
    {"real": main_real, "shuffle": run_shuffle_test, "width": run_width_test, 
     "halves": run_halves, "excl": run_excl_robustness}[mode]()
    """
    df = pd.read_csv(PAIRS_FILE)
    dps_map = hp.read_map(DPS_MAP_FILE)
    dps_map = dps_map - np.mean(dps_map)
    res = run_joint_analysis(dps_map, df)

    for name, data in res.items():
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        tag = f'tomo{TOMO_BIN}_{clean_name}'
        excl = JOINT_CONFIGS[name].get('excl', (0.5, 0.3))

        _save_map(data['stack'], f'{OUTPUT_DIR}/dps_true_stack_{tag}.png', f'Stacked DPS ({name})', r'DPS $\kappa$')
        _save_map(data['model'], f'{OUTPUT_DIR}/dps_null_total_{tag}.png', f'Joint 2-halo model ({name})', r'DPS $\kappa$')
        _save_filament_map(data['filament'], f'{OUTPUT_DIR}/dps_filament_map_{tag}.png', f'Subtracted map ({name})', r'DPS $\kappa$', excl=excl)
        plot_transverse_profile(data['profile'], f'{OUTPUT_DIR}/dps_transverse_profile_{tag}.png', f'1D Filament Profile ({name})')

        np.savez(f'{OUTPUT_DIR}/maps_{tag}.npz', stack=data['stack'], model=data['model'],
                 filament=data['filament'], profile_y=data['profile']['y'],
                 profile_mean=data['profile']['mean'], profile_sigma=data['profile']['sigma'])
    """
if __name__ == "__main__":
    main()

