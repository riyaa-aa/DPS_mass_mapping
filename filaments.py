# cleaned up version of get_filament_stack.py
import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time
import pandas as pd

# configuration
TOMO_BIN = 4
DPS_MAP_FILE = f"/home2/supranta/PosteriorSampling/denoising_diffusion_pytorch/denoising_diffusion_pytorch/results_desy3_big/DPS_healpix_maps/dps_kappa_hp_tomo{TOMO_BIN}.fits"
PAIRS_FILE = 'redmapper_cluster_pairs.csv'
#PAIRS_FILE = 'redmapper_cluster_pairs_shuffled.csv'
OUTPUT_DIR = 'filaments'
FILE_SUFFIX = "_joint"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_MIN, GRID_MAX, GRID_RES = -2.0, 2.0, 200
GAUSSIAN_SMOOTH_KERNEL = 0.05
X_LIN = np.linspace(GRID_MIN, GRID_MAX, GRID_RES)
Y_LIN = np.linspace(GRID_MIN, GRID_MAX, GRID_RES)
X_GRID, Y_GRID = np.meshgrid(X_LIN, Y_LIN)
PIXEL_WIDTH = (GRID_MAX - GRID_MIN) / (GRID_RES - 1)
BOX_MASK = (X_GRID >= -0.3) & (X_GRID <= 0.3) & (Y_GRID >= -0.15) & (Y_GRID <= 0.15)

JOINT_CONFIGS = {"Joint fit": dict(m2=False), "Joint fit + m2": dict(m2=True)}
EXCL = (0.5, 0.3)

# stacking
def _smooth(signal):
    return gaussian_filter(signal, sigma=GAUSSIAN_SMOOTH_KERNEL / PIXEL_WIDTH)

def _theta_for_row(row):
    ra_L, dec_L = np.radians(row.ra_L), np.radians(row.dec_L)
    ra_R, dec_R = np.radians(row.ra_R), np.radians(row.dec_R)
    dra = (ra_R - ra_L + np.pi) % (2 * np.pi) - np.pi

    alpha_mid = (ra_L + 0.5 * dra) % (2 * np.pi)
    delta_mid = 0.5 * (dec_L + dec_R)

    dra_L = (ra_L - alpha_mid + np.pi) % (2 * np.pi) - np.pi
    dra_R = (ra_R - alpha_mid + np.pi) % (2 * np.pi) - np.pi

    X_L, Y_L = -dra_L * np.cos(delta_mid), dec_L - delta_mid
    X_R, Y_R = -dra_R * np.cos(delta_mid), dec_R - delta_mid

    return np.arctan2(Y_R - Y_L, X_R - X_L)

def stack_true_filament(dps_map, df, smooth=True, weights=None):
    n = len(df)
    sum_map = np.zeros((GRID_RES, GRID_RES))
    box_k = np.zeros(n)
    pair_stats = np.zeros(n)

    for i, row in enumerate(df.itertuples()):
        ra_L, dec_L = np.radians(row.ra_L), np.radians(row.dec_L)
        ra_R, dec_R = np.radians(row.ra_R), np.radians(row.dec_R)
        dra = (ra_R - ra_L + np.pi) % (2 * np.pi) - np.pi

        alpha_mid = (ra_L + 0.5 * dra) % (2 * np.pi)
        delta_mid = 0.5 * (dec_L + dec_R)
        theta = _theta_for_row(row)

        Xs, Ys = X_GRID * row.sep_rad, Y_GRID * row.sep_rad
        Xp = Xs * np.cos(theta) - Ys * np.sin(theta)
        Yp = Xs * np.sin(theta) + Ys * np.cos(theta)

        alpha_eval = alpha_mid - Xp / np.cos(delta_mid)
        delta_eval = delta_mid + Yp
        
        pair_map = hp.get_interp_val(dps_map, np.pi/2 - delta_eval, alpha_eval)

        if smooth:
            pair_map = _smooth(pair_map)

        sum_map += pair_map
        box_k[i] = np.mean(pair_map[BOX_MASK])
        if weights is not None:
            pair_stats[i] = np.sum(weights * pair_map)

    avg = sum_map / n

    if weights is None:
        return avg, box_k
    return avg, box_k, pair_stats

# joint-fit null
def _joint_design(n_nodes=40, n_nodes_m2=12, r_max=3.3, excl=EXCL, m2=True, ridge=1e-8):
    npix = GRID_RES * GRID_RES
    rows = np.arange(npix)

    def basis(xc, nodes, weight=None):
        R = np.hypot(X_GRID - xc, Y_GRID).ravel()   # dist of each pixel from the cluster
        idx = np.clip(np.searchsorted(nodes, R) - 1, 0, len(nodes) - 2)     # index of the node just below R
        t = np.clip((R - nodes[idx]) / (nodes[idx + 1] - nodes[idx]), 0, 1) # fraction of the way to the next node
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

    blocks = [basis(-0.5, r0), basis(+0.5, r0)]     # left and right profiles

    if m2:
        for xc in (-0.5, +0.5):
            phi = np.arctan2(Y_GRID, X_GRID - xc).ravel()
            blocks.append(basis(xc, r2, weight=np.cos(2*phi)))  # quadrupole term profiles

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
    box = BOX_MASK if box is None else box
    A, fit_mask, ridge = _joint_design(**kw)
    Af = A[fit_mask]
    b = box.ravel().astype(float)
    b /= b.sum()    # averages over the box
    G = Af.T @ Af + ridge * np.eye(A.shape[1])
    u = np.zeros(A.shape[0])
    u[fit_mask] = Af @ np.linalg.solve(G, A.T @ b)  # 0 inside strip
    return (b - u).reshape(GRID_RES, GRID_RES)

# measurement and errors
def measure_filament_box(filament_map, x_range=(-0.3, 0.3), y_range=(-0.15, 0.15)):
    m = ((X_GRID >= x_range[0]) & (X_GRID <= x_range[1]) &
         (Y_GRID >= y_range[0]) & (Y_GRID <= y_range[1]))
    return np.mean(filament_map[m]), np.sum(filament_map[m]) * PIXEL_WIDTH**2

def compute_bootstrap_stats(per_pair, n_bootstraps=1000, seed=42):
    per_pair = np.asarray(per_pair)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(per_pair), size=(n_bootstraps, len(per_pair)), replace=True)
    boots = per_pair[idx].mean(axis=1)
    mean, sigma = boots.mean(), boots.std(ddof=1)
    return {"mean": mean, "sigma": sigma, "snr": mean / sigma if sigma > 0 else 0.0,
            "ci_68": (np.percentile(boots, 15.85), np.percentile(boots, 84.15))}

def run_joint_analysis(dps_map, df):
    out = {}
    for name, kw in JOINT_CONFIGS.items():
        W = joint_box_weights(**kw)
        stack, _, stats = stack_true_filament(dps_map, df, weights=W)
        model = fit_two_halo_model(stack, **kw)
        direct = measure_filament_box(stack - model)[0]
        assert np.isclose(direct, stats.mean(), rtol=1e-6, atol=1e-10), (direct, stats.mean())
        out[name] = dict(stack=stack, model=model, filament=stack - model,
                         boot=compute_bootstrap_stats(stats), stats=stats)
        b = out[name]['boot']
        print(f"{name:16s} box = {direct:+.4e}  sigma = {b['sigma']:.2e}  S/N = {b['snr']:.2f}")
    return out

# plotting
def _save_map(data, filename, title, cbar_label):
    plt.figure(figsize=(8, 7))
    vmin, vmax = np.percentile(data, [1, 99.5])
    im = plt.imshow(data, cmap='viridis', origin='lower',
                    extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX], vmin=vmin, vmax=vmax)
    plt.colorbar(im, pad=0.02).set_label(cbar_label, fontsize=12)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved to {filename}")
    
def _save_filament_map(data, filename, title, cbar_label, excl=EXCL):
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

# main
def main():
    t0 = time.perf_counter()
    df = pd.read_csv(PAIRS_FILE)
    dps_map = hp.read_map(DPS_MAP_FILE)
    dps_map = dps_map - np.mean(dps_map)
    
    
    # testing
    Wl = joint_box_weights(box=BOX_MASK & (X_GRID < 0), m2=True)
    Wr = joint_box_weights(box=BOX_MASK & (X_GRID > 0), m2=True)
    _, _, st = stack_true_filament(dps_map, df, weights=Wr - Wl)   # per-pair (right - left)
    b = compute_bootstrap_stats(st); print(b['mean'], b['sigma'], b['snr'])
    p = np.random.default_rng(0).permutation(len(st)); h = len(st)//2
    print(st[p[:h]].mean(), st[p[h:]].mean())   # a real effect should have the same sign in both 

    """
    # creating & plotting the maps    

    res = run_joint_analysis(dps_map, df)
    m2 = res["Joint fit + m2"]
    m0 = res["Joint fit"]
    tag = f'tomo{TOMO_BIN}{FILE_SUFFIX}'
    _save_map(m2['stack'], f'{OUTPUT_DIR}/dps_true_stack_{tag}.png', 
              f'Stacked DPS (Tomo {TOMO_BIN})', r'DPS $\kappa$')
    _save_map(m2['model'], f'{OUTPUT_DIR}/dps_null_total_{tag}.png', 
              f'Joint 2-halo model (Tomo {TOMO_BIN})', r'DPS $\kappa$')
    _save_filament_map(m2['filament'], f'{OUTPUT_DIR}/dps_filament_map_{tag}.png', 
                       f'Subtracted map (Tomo {TOMO_BIN})', r'DPS $\kappa$')
    np.savez(f'{OUTPUT_DIR}/maps_{tag}.npz', stack=m2['stack'], model=m2['model'], filament=m2['filament'])
    """
    print(f"Total time: {(time.perf_counter()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()


