import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_rgba

# ============================================================
# Config
# ============================================================
N_CLUSTERS = 3  # number of GMM components

# ============================================================
# Load data
# ============================================================
data = np.load('mcmc_samples_no_pressure_south.npz')
samples = data['flat_samples']  # (n_steps, 5)
print(f'Loaded {samples.shape[0]} samples, {samples.shape[1]} parameters')

param_labels = [
    'z [AU]',
    r'$v_r$ [km/s]',
    r'$\log_{10}(\omega)$ [round/yr]',
    r'$\theta_\mathrm{axis}$ [deg]',
    r'$\phi_\mathrm{axis}$ [deg]',
]

# ============================================================
# Thin samples for clustering (use 50000 to keep GMM fast)
# ============================================================
rng = np.random.default_rng(42)
n_thin = min(50000, samples.shape[0])
idx_thin = rng.choice(samples.shape[0], size=n_thin, replace=False)
samples_thin = samples[idx_thin]

# ============================================================
# GMM clustering
# ============================================================
print(f'Running GMM with {N_CLUSTERS} components...')
gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42, n_init=10)
labels_thin = gmm.fit_predict(samples_thin)

# Assign full sample labels via nearest GMM component
probs_full = gmm.predict_proba(samples)
labels_full = np.argmax(probs_full, axis=1)

clusters = []
for k in range(N_CLUSTERS):
    clusters.append(samples[labels_full == k])

# Sort clusters by size descending
order = np.argsort([-len(c) for c in clusters])
clusters = [clusters[i] for i in order]
# Re-label so largest is cluster 0, etc.
old_to_new = {old: new for new, old in enumerate(order)}
labels_full = np.array([old_to_new[l] for l in labels_full])

cluster_names = [f'Cluster {i}' for i in range(N_CLUSTERS)]
for i, cl in enumerate(clusters):
    print(f'{cluster_names[i]}: {cl.shape[0]} samples ({100 * cl.shape[0] / samples.shape[0]:.1f}%)')

# ============================================================
# Color palette
# ============================================================
cmap = cm.get_cmap('tab10')
colors = [cmap(i % 10) for i in range(N_CLUSTERS)]

# ============================================================
# Corner plot with smoothed 2D histograms
# ============================================================
n_params = samples.shape[1]
fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))

for i in range(n_params):
    for j in range(n_params):
        ax = axes[i, j]

        if i == j:
            # --- Diagonal: 1D histogram (line, not filled) ---
            for clr, name, cl in zip(colors, cluster_names, clusters):
                ax.hist(cl[:, i], bins=40, histtype='step', color=clr,
                        linewidth=1.5, density=True, label=name)
            ax.set_xlim(samples[:, i].min(), samples[:, i].max())
            ax.set_yticks([])
            if i == 0:
                ax.legend(fontsize=7, loc='upper right')
            if i == n_params - 1:
                ax.set_xlabel(param_labels[i], fontsize=8)
            else:
                ax.set_xticklabels([])

        elif j < i:
            # --- Lower triangle: smoothed 2D density via hexbin ---
            for clr, cl in zip(colors, clusters):
                clr_rgb = to_rgba(clr)[:3]
                cmap_cl = plt.cm.colors.LinearSegmentedColormap.from_list(
                    '', [(1, 1, 1), clr_rgb])
                ax.hexbin(cl[:, j], cl[:, i], gridsize=30,
                          cmap=cmap_cl,
                          mincnt=1, bins='log', alpha=0.7,
                          extent=[samples[:, j].min(), samples[:, j].max(),
                                  samples[:, i].min(), samples[:, i].max()])
            if j == 0:
                ax.set_ylabel(param_labels[i], fontsize=8)
            else:
                ax.set_yticklabels([])
            if i == n_params - 1:
                ax.set_xlabel(param_labels[j], fontsize=8)
            else:
                ax.set_xticklabels([])

        else:
            # Upper triangle: hide
            ax.set_visible(False)

fig.suptitle(f'MCMC Samples — GMM Clustering ({N_CLUSTERS} components)', fontsize=13, y=0.98)
plt.tight_layout()
fig.savefig('mcmc_clustered_nop_corner.png', dpi=150, bbox_inches='tight')
print('Saved mcmc_clustered_nop_corner.png')
plt.close(fig)

# ============================================================
# Statistics per cluster
# ============================================================
print('\n' + '=' * 70)
print('Cluster Statistics (median, 16th, 84th percentiles)')
print('=' * 70)

for cl_name, cl in zip(cluster_names, clusters):
    q = np.percentile(cl, [16, 50, 84], axis=0)
    print(f'\n--- {cl_name} ({cl.shape[0]} samples) ---')
    for i, label in enumerate(param_labels):
        val_50 = q[1, i]
        val_16 = q[0, i]
        val_84 = q[2, i]
        print(f'  {label:>35s}: {val_50:12.4f}   (-{val_50 - val_16:.4f} / +{val_84 - val_50:.4f})')

# Save cluster labels alongside samples
save_kwargs = dict(flat_samples=samples, cluster_labels=labels_full)
for i, cl in enumerate(clusters):
    save_kwargs[f'cluster{i}'] = cl
np.savez('mcmc_samples_clustered_nop.npz', **save_kwargs)
print('\nSaved mcmc_samples_clustered_nop.npz')
