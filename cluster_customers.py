import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load
customers = pd.read_csv('customers_clean.csv')

# ============================================
# 1. PICK FEATURES FOR CLUSTERING
# ============================================

# We'll cluster on spending behavior only
features = customers[['avg_spend', 'purchase_frequency']].copy()

# Scale (KMeans needs this or avg_spend dominates)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ============================================
# 2. FIND BEST K (number of clusters)
# ============================================

# Elbow method: try K=1 to 10, pick where curve bends
inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(features_scaled)
    inertias.append(km.inertia_)

print("Elbow scores (lower = tighter clusters):")
for k, inertia in zip(K_range, inertias):
    print(f"  K={k}: {inertia:.0f}")

# ============================================
# 3. TRAIN WITH K=3 (we know there are 3 segments, but model doesn't)
# ============================================

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customers['cluster'] = kmeans.fit_predict(features_scaled)

# ============================================
# 4. SEE WHAT EACH CLUSTER LOOKS LIKE
# ============================================

print("\nCluster profiles:")
for c in range(3):
    cluster_data = customers[customers['cluster'] == c]
    print(f"\nCluster {c} ({len(cluster_data)} customers):")
    print(f"  Avg spend: ${cluster_data['avg_spend'].mean():.0f}")
    print(f"  Avg frequency: {cluster_data['purchase_frequency'].mean():.1f}")

# ============================================
# 5. COMPARE TO HIDDEN TRUTH
# ============================================

# We secretly know the real segments from data generation
# Let's see if clusters match
truth = customers.groupby('segment').agg(
    count=('customer_id', 'count'),
    avg_spend=('avg_spend', 'mean'),
    avg_freq=('purchase_frequency', 'mean')
)
print("\nGround truth (hidden from model):")
print(truth)

# ============================================
# 6. SAVE
# ============================================

customers.to_csv('customers_clustered.csv', index=False)
print("\nSaved customers_clustered.csv")