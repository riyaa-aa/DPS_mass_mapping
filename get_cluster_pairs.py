import h5py
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import healpy as hp
from healpy import ang2pix, read_map

# 1. Configuration
REDMAPPER_FILE = 'y3_redmapper_v6.4.22+2_release.h5'
OUTPUT_PAIRS_FILE = 'redmapper_cluster_pairs_shuffled.csv'

# Cosmology (H0=100 so that distances natively return in h^-1 Mpc)
cosmo = FlatLambdaCDM(H0=100, Om0=0.3)

def find_cluster_pairs():
    print("Loading redMaPPer catalog...")
    with h5py.File(REDMAPPER_FILE, 'r') as f:
        ra = f['catalog/cluster/ra'][:]
        dec = f['catalog/cluster/dec'][:]
        z = f['catalog/cluster/z_lambda'][:]
        lam = f['catalog/cluster/lambda_chisq'][:]

    # Filter 1: Richness threshold
    valid_mask = (lam > 20) & (z >= 0.1) & (z <= 0.65)
    ra, dec, z, lam = ra[valid_mask], dec[valid_mask], z[valid_mask], lam[valid_mask]
    
    print(f"Clusters remaining after lambda > 20 cut: {len(ra)}")

    # Create SkyCoord objects for fast searching
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    
    # 6 degrees is a safe upper bound for 25 h^-1 Mpc at z ~ 0.1
    print("Searching for neighbors within 6 degrees...")
    idx1, idx2, sep2d, _ = coords.search_around_sky(coords, 6*u.deg)

    # Filter out self-matches and duplicate pairs (keep idx1 < idx2)
    unique_pairs = idx1 < idx2
    idx1, idx2, sep_rad = idx1[unique_pairs], idx2[unique_pairs], sep2d[unique_pairs].radian

    # Filter 2: Redshift difference Delta_z < 0.01
    dz = np.abs(z[idx1] - z[idx2])
    z_mask = dz < 0.01
    idx1, idx2, sep_rad = idx1[z_mask], idx2[z_mask], sep_rad[z_mask]

    # Filter 3: Comoving physical separation [15, 25] h^-1 Mpc
    mean_z = (z[idx1] + z[idx2]) / 2.0
    
    # Use comoving transverse distance, NOT angular diameter distance
    D_M = cosmo.comoving_transverse_distance(mean_z).value # Returns h^-1 Mpc
    comoving_sep = sep_rad * D_M
    
    #D_A = cosmo.angular_diameter_distance(mean_z).value # Returns h^-1 Mpc
    #proper_sep = sep_rad * D_A

    dist_mask = (comoving_sep >= 15) & (comoving_sep <= 25)
    #dist_mask = (proper_sep >= 15) & (proper_sep <= 25)
    idx1, idx2 = idx1[dist_mask], idx2[dist_mask]
    sep_rad = sep_rad[dist_mask]
    
    print(f"Found {len(idx1)} valid cluster pairs.")

    lam1, lam2 = lam[idx1], lam[idx2]
    mask_1_left = np.random.rand(len(idx1)) > 0.5

    # Create arrays to hold the randomized pairs
    ra_L, dec_L, lam_L = np.zeros_like(ra[idx1]), np.zeros_like(dec[idx1]), np.zeros_like(lam[idx1])
    ra_R, dec_R, lam_R = np.zeros_like(ra[idx1]), np.zeros_like(dec[idx1]), np.zeros_like(lam[idx1])

    # If random draw is True, cluster 1 becomes Left
    ra_L[mask_1_left], dec_L[mask_1_left], lam_L[mask_1_left] = ra[idx1][mask_1_left], dec[idx1][mask_1_left], lam1[mask_1_left]
    ra_R[mask_1_left], dec_R[mask_1_left], lam_R[mask_1_left] = ra[idx2][mask_1_left], dec[idx2][mask_1_left], lam2[mask_1_left]

    # Otherwise, cluster 2 becomes Left
    mask_2_left = ~mask_1_left
    ra_L[mask_2_left], dec_L[mask_2_left], lam_L[mask_2_left] = ra[idx2][mask_2_left], dec[idx2][mask_2_left], lam2[mask_2_left]
    ra_R[mask_2_left], dec_R[mask_2_left], lam_R[mask_2_left] = ra[idx1][mask_2_left], dec[idx1][mask_2_left], lam1[mask_2_left]

    z_L = np.where(mask_1_left, z[idx1], z[idx2])
    z_R = np.where(mask_1_left, z[idx2], z[idx1])

    '''
    # Assign "Left" (richer) and "Right" (less rich) based on the paper's stacking method
    lam1, lam2 = lam[idx1], lam[idx2]

    # Create arrays to hold the sorted pairsi
    ra_L, dec_L, lam_L = np.zeros_like(ra[idx1]), np.zeros_like(dec[idx1]), np.zeros_like(lam[idx1])
    ra_R, dec_R, lam_R = np.zeros_like(ra[idx1]), np.zeros_like(dec[idx1]), np.zeros_like(lam[idx1])

    # If cluster 1 is richer, it becomes Left
    mask_1_richer = lam1 >= lam2
    ra_L[mask_1_richer], dec_L[mask_1_richer], lam_L[mask_1_richer] = ra[idx1][mask_1_richer], dec[idx1][mask_1_richer], lam1[mask_1_richer]
    ra_R[mask_1_richer], dec_R[mask_1_richer], lam_R[mask_1_richer] = ra[idx2][mask_1_richer], dec[idx2][mask_1_richer], lam2[mask_1_richer]

    # If cluster 2 is richer, it becomes Left
    mask_2_richer = ~mask_1_richer
    ra_L[mask_2_richer], dec_L[mask_2_richer], lam_L[mask_2_richer] = ra[idx2][mask_2_richer], dec[idx2][mask_2_richer], lam2[mask_2_richer]
    ra_R[mask_2_richer], dec_R[mask_2_richer], lam_R[mask_2_richer] = ra[idx1][mask_2_richer], dec[idx1][mask_2_richer], lam1[mask_2_richer]

    z_L = np.where(mask_1_richer, z[idx1], z[idx2])
    z_R = np.where(mask_1_richer, z[idx2], z[idx1])
    '''


    df = pd.DataFrame({
    'ra_L': ra_L, 'dec_L': dec_L, 'z_L': z_L, 'lam_L': lam_L,
    'ra_R': ra_R, 'dec_R': dec_R, 'z_R': z_R, 'lam_R': lam_R,
    'sep_rad': sep_rad,
    'sep_hmpc': sep_rad * cosmo.comoving_transverse_distance(
        (z_L + z_R) / 2.0).value
    })

    mask = read_map('glimpse_mask.fits', verbose=False)
    nside = hp.get_nside(mask)

    def in_mask(ra_arr, dec_arr):
        pix = ang2pix(nside, np.radians(90 - dec_arr), np.radians(ra_arr))
        return mask[pix] == 1

    valid_pairs = in_mask(ra_L, dec_L) & in_mask(ra_R, dec_R)
    df = df[valid_pairs]

    df.to_csv(OUTPUT_PAIRS_FILE, index=False)
    print(f"Saved {len(df)} pairs to {OUTPUT_PAIRS_FILE}")

if __name__ == "__main__":
    find_cluster_pairs()
