import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load
customers = pd.read_csv('customers_clustered.csv')

# ============================================
# 1. PCA: 2 FEATURES → 2D POINTS
# ============================================

features = customers[['avg_spend', 'purchase_frequency']]
features_scaled = StandardScaler().fit_transform(features)

pca = PCA(n_components=2)
points_2d = pca.fit_transform(features_scaled)

# ============================================
# 2. PLOT
# ============================================

plt.figure(figsize=(8, 6))

colors = ['#e74c3c', '#3498db', '#2ecc71']
labels = ['Premium', 'Regular', 'Budget']

for cluster_id in [0, 1, 2]:
    mask = customers['cluster'] == cluster_id
    plt.scatter(
        points_2d[mask, 0],
        points_2d[mask, 1],
        c=colors[cluster_id],
        label=f'Cluster {cluster_id}',
        alpha=0.6,
        s=20
    )

plt.xlabel('PC1 (spend + frequency combined)')
plt.ylabel('PC2 (remaining variation)')
plt.title('Customer Clusters')
plt.legend()
plt.tight_layout()
plt.savefig('cluster_plot.png', dpi=150)
print("Saved cluster_plot.png")