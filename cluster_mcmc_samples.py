import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import to_rgba

# ============================================================
# Load data
# ============================================================
data = np.load('mcmc_samples_with_pressure_south.npz')
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
# GMM clustering (2 components)
# ============================================================
print('Running GMM with 3 components...')
gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
labels_thin = gmm.fit_predict(samples_thin)

# Assign full sample labels via nearest GMM component
probs_full = gmm.predict_proba(samples)
labels_full = np.argmax(probs_full, axis=1)

cluster0 = samples[labels_full == 0]
cluster1 = samples[labels_full == 1]

# Ensure cluster0 is the larger one (arbitrary but consistent labeling)
if len(cluster0) < len(cluster1):
    cluster0, cluster1 = cluster1, cluster0
    labels_full = 1 - labels_full

print(f'Cluster 0: {cluster0.shape[0]} samples ({100*cluster0.shape[0]/samples.shape[0]:.1f}%)')
print(f'Cluster 1: {cluster1.shape[0]} samples ({100*cluster1.shape[0]/samples.shape[0]:.1f}%)')

# ============================================================
# Corner plot with smoothed 2D histograms
# ============================================================
n_params = samples.shape[1]
fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))

colors = ['#1f77b4', '#d62728']  # blue, red
cluster_names = ['Cluster 0', 'Cluster 1']
alpha_2d = 0.6

for i in range(n_params):
    for j in range(n_params):
        ax = axes[i, j]

        if i == j:
            # --- Diagonal: 1D histogram (line, not filled) ---
            for clr, label, cl in zip(colors, cluster_names, [cluster0, cluster1]):
                ax.hist(cl[:, i], bins=40, histtype='step', color=clr,
                        linewidth=1.5, density=True, label=label)
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
            for clr, cl in zip(colors, [cluster0, cluster1]):
                rgba = to_rgba(clr, alpha_2d)
                hb = ax.hexbin(cl[:, j], cl[:, i], gridsize=30,
                               cmap=plt.cm.Blues if clr == colors[0] else plt.cm.Reds,
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

fig.suptitle('MCMC Samples — GMM Clustering (2 components)', fontsize=13, y=0.98)
plt.tight_layout()
fig.savefig('mcmc_clustered_corner.png', dpi=150, bbox_inches='tight')
print('Saved mcmc_clustered_corner.png')
plt.close(fig)

# ============================================================
# Statistics per cluster
# ============================================================
print('\n' + '=' * 70)
print('Cluster Statistics (median, 16th, 84th percentiles)')
print('=' * 70)

for cl_name, cl in zip(cluster_names, [cluster0, cluster1]):
    q = np.percentile(cl, [16, 50, 84], axis=0)
    print(f'\n--- {cl_name} ({cl.shape[0]} samples) ---')
    for i, label in enumerate(param_labels):
        val_50 = q[1, i]
        val_16 = q[0, i]
        val_84 = q[2, i]
        print(f'  {label:>35s}: {val_50:12.4f}   (-{val_50 - val_16:.4f} / +{val_84 - val_50:.4f})')

# Save cluster labels alongside samples
np.savez('mcmc_samples_clustered.npz',
         flat_samples=samples,
         cluster_labels=labels_full,
         cluster0=cluster0,
         cluster1=cluster1)
print('\nSaved mcmc_samples_clustered.npz')
