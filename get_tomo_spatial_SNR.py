import time
import h5py
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

def get_single_map_noise(map_data, mask_map, n_samples=10000, cutout_size_arcmin=30, res_arcmin=1.0):
    nside = hp.get_nside(mask_map)
    valid_indices = np.where(mask_map == 1)[0]
    
    if len(valid_indices) < n_samples:
        n_samples = len(valid_indices)

    random_pixel_indices = np.random.choice(valid_indices, size=n_samples, replace=False)
    theta, phi = hp.pix2ang(nside, random_pixel_indices)
    ra_list = np.degrees(phi)
    dec_list = 90.0 - np.degrees(theta)

    xsize = int(cutout_size_arcmin * 2 / res_arcmin)
    stacked_noise = np.zeros((xsize, xsize))
    stacked_sq_noise = np.zeros((xsize, xsize))

    for i in range(n_samples):
        cutout = hp.gnomview(
            map_data, rot=(ra_list[i], dec_list[i], 0), reso=res_arcmin,
            xsize=xsize, return_projected_map=True, no_plot=True
        )
        clean_cutout = np.nan_to_num(cutout)
        stacked_noise += clean_cutout
        stacked_sq_noise += clean_cutout**2

    pop_mean = stacked_noise / n_samples
    pop_variance = np.maximum((stacked_sq_noise / n_samples) - (pop_mean**2), 0)
    pop_std = np.sqrt(pop_variance)

    return pop_mean, pop_std

def stack_signal(map_data, ra, dec, cutout_size_arcmin=30, res_arcmin=1.0):
    n_clusters = len(ra)
    xsize = int(cutout_size_arcmin * 2 / res_arcmin)
    stacked_map = np.zeros((xsize, xsize))

    for i in range(n_clusters):
        cutout = hp.gnomview(
            map_data, rot=(ra[i], dec[i], 0), reso=res_arcmin,
            xsize=xsize, return_projected_map=True, no_plot=True
        )
        stacked_map += np.nan_to_num(cutout)

    return stacked_map / n_clusters

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    start_time = time.time()

    # --- CONFIGURATION ---
    TARGET_BIN = 4  # Which bin to visualize (1, 2, 3, or 4)
    CUTOUT_SIZE = 30
    RESOLUTION = 1.0

    mask_file = 'glimpse_mask.fits'
    redmapper_file = 'y3_redmapper_v6.4.22+2_release.h5'
    bnt_matrix_file = 'desy3_kernels.h5' 

    print(f"Generating Side-by-Side Spatial SNR Map for DPS - Bin {TARGET_BIN}...")

    # 1. Load Data
    mask_map = hp.read_map(mask_file, verbose=False)
    
    with h5py.File(redmapper_file, 'r') as f:
        ra_full = f['catalog/cluster/ra'][:]
        dec_full = f['catalog/cluster/dec'][:]
        richness_full = f['catalog/cluster/lambda_chisq'][:]
        redshift_full = f['catalog/cluster/z_lambda'][:]

    with h5py.File(bnt_matrix_file, 'r') as f:
        nulling_matrix = f['nulling_matrix'][:] 
        z_kernel = f['kernel_z/z'][:]
        nulled_q_z = f['kernel_z/nulled_q_z'][:]

    # 2. Load all 4 raw DPS maps
    kappa_raw_list = []
    for t in [1, 2, 3, 4]:
        map_file = f"/home2/supranta/PosteriorSampling/denoising_diffusion_pytorch/denoising_diffusion_pytorch/results_desy3_big/DPS_healpix_maps/dps_kappa_hp_tomo{t}.fits"
        kappa_raw_list.append(hp.read_map(map_file, verbose=False))
        
    kappa_raw = np.array(kappa_raw_list) 
    
    # 3. Apply BNT Transform
    print("Applying BNT Transform...")
    kappa_tilde = nulling_matrix @ kappa_raw 
    bin_idx = TARGET_BIN - 1

    # ==========================================
    # PIPELINE A: RAW TOMOGRAPHY (NO BNT)
    # ==========================================
    print(f"\nProcessing Raw Tomography for Bin {TARGET_BIN}...")
    map_data_raw = kappa_raw[bin_idx]
    
    # Use standard richness filter (no redshift cut)
    mask_raw = (richness_full > 20)
    ra_raw = ra_full[mask_raw]
    dec_raw = dec_full[mask_raw]
    print(f"Stacking {len(ra_raw)} clusters on raw map...")

    pop_mean_raw, pop_std_raw = get_single_map_noise(
        map_data_raw, mask_map, n_samples=10000, 
        cutout_size_arcmin=CUTOUT_SIZE, res_arcmin=RESOLUTION
    )
    signal_stack_raw = stack_signal(
        map_data_raw, ra_raw, dec_raw, 
        cutout_size_arcmin=CUTOUT_SIZE, res_arcmin=RESOLUTION
    )
    
    snr_map_raw = (signal_stack_raw - pop_mean_raw) / pop_std_raw

    # ==========================================
    # PIPELINE B: BNT TRANSFORMED
    # ==========================================
    print(f"\nProcessing BNT Transformed Map for Bin {TARGET_BIN}...")
    map_data_bnt = kappa_tilde[bin_idx]

    # Calculate dynamic 5% redshift filter
    kernel_max = np.max(nulled_q_z[bin_idx])
    active_indices = np.where(nulled_q_z[bin_idx] > 0.05 * kernel_max)[0]
    z_min = z_kernel[active_indices[0]] if len(active_indices) > 0 else 0.0
    
    # Apply joint mask: richness > 20 AND redshift >= z_min
    mask_bnt = (richness_full > 20) & (redshift_full >= z_min)
    ra_bnt = ra_full[mask_bnt]
    dec_bnt = dec_full[mask_bnt]
    print(f"Kernel active at z >= {z_min:.3f} | Stacking {len(ra_bnt)} optimized clusters...")

    pop_mean_bnt, pop_std_bnt = get_single_map_noise(
        map_data_bnt, mask_map, n_samples=10000, 
        cutout_size_arcmin=CUTOUT_SIZE, res_arcmin=RESOLUTION
    )
    signal_stack_bnt = stack_signal(
        map_data_bnt, ra_bnt, dec_bnt, 
        cutout_size_arcmin=CUTOUT_SIZE, res_arcmin=RESOLUTION
    )
    
    snr_map_bnt = (signal_stack_bnt - pop_mean_bnt) / pop_std_bnt

    # ==========================================
    # PLOTTING: 1x2 COMPARISON
    # ==========================================
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Raw Map Plot
    im_raw = axes[0].imshow(snr_map_raw, cmap='RdBu_r', origin='upper')
    cbar_raw = fig.colorbar(im_raw, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_raw.set_label('SNR')
    axes[0].set_title(f'Raw Tomography (DPS Tomo {TARGET_BIN})')
    
    # BNT Map Plot
    im_bnt = axes[1].imshow(snr_map_bnt, cmap='RdBu_r', origin='upper')
    cbar_bnt = fig.colorbar(im_bnt, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_bnt.set_label('SNR')
    axes[1].set_title(f'BNT Transformed + Filtered (DPS BNT {TARGET_BIN})')
    
    plt.tight_layout()
    plt.savefig(f'snr_heatmaps/dps_spatial_snr_comparison_bin{TARGET_BIN}.png', dpi=300)
    print(f"Total Execution Time: {(time.time() - start_time)/60:.2f} minutes.")
    plt.show()
