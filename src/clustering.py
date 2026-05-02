"""
Social Media User Behavior Clustering (Behavior-Based Segmentation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = r"D:\social_media_clustering_project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "instagram_usage_lifestyle.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

print("Dataset loaded:", df.shape)
print("Columns:", df.columns.tolist())

# ============================================================
# FEATURE ENGINEERING (BEHAVIOR-BASED)
# ============================================================

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Avoid division by zero
epsilon = 1

# Behavior features
df['creator_ratio'] = df['posts_created_per_week'] / (df['sessions_per_day'] + epsilon)

df['interaction_ratio'] = (
    df['likes_given_per_day'] + df['comments_written_per_day']
) / (df['sessions_per_day'] + epsilon)

df['social_ratio'] = df['dms_sent_per_week'] / (df['sessions_per_day'] + epsilon)

df['reel_preference'] = df['time_on_reels_per_day'] / (
    df['time_on_feed_per_day'] + df['time_on_reels_per_day'] + epsilon
)

df['scroll_ratio'] = df['time_on_feed_per_day'] / (
    df['daily_active_minutes_instagram'] + epsilon
)

df['avg_session_intensity'] = df['daily_active_minutes_instagram'] / (
    df['sessions_per_day'] + epsilon
)

# ============================================================
# FINAL FEATURES (BEHAVIOR ONLY)
# ============================================================

FEATURES = [
    'age',
    'creator_ratio',
    'interaction_ratio',
    'social_ratio',
    'reel_preference',
    'scroll_ratio',
    'avg_session_intensity'
]

print(f"\nUsing {len(FEATURES)} behavior-based features")

# ============================================================
# SAMPLE FOR SPEED
# ============================================================

sample_size = min(50000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

# Fill missing values
df_sample[FEATURES] = df_sample[FEATURES].fillna(df_sample[FEATURES].median())

X_sample = df_sample[FEATURES]

# Scale
scaler_sample = StandardScaler()
X_scaled_sample = scaler_sample.fit_transform(X_sample)

print("Sample prepared")

# ============================================================
# FORCE 4 CLUSTERS (BEHAVIOR SEGMENTS)
# ============================================================

K = 4
print(f"\nUsing K = {K} for behavior segmentation")

# ============================================================
# TRAIN ON FULL DATA
# ============================================================

df_full = df.copy()
df_full[FEATURES] = df_full[FEATURES].fillna(df_full[FEATURES].median())

X_full = df_full[FEATURES]

scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(X_full)

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df_full['cluster'] = kmeans.fit_predict(X_scaled_full)

print("\nCluster distribution:")
print(df_full['cluster'].value_counts())

# ============================================================
# PCA VISUALIZATION
# ============================================================

print("\nGenerating PCA plot...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_full)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_full['cluster'], alpha=0.5)
plt.title("Behavior-Based User Clusters (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plot_path = os.path.join(OUTPUT_PATH, "behavior_clusters.png")
plt.savefig(plot_path, dpi=100)
plt.close()

print(f"PCA plot saved at: {plot_path}")

# ============================================================
# CLUSTER INTERPRETATION
# ============================================================

print("\nCluster Profiles:\n")

cluster_names = {}
global_means = df_full[FEATURES].mean()

for cluster in sorted(df_full['cluster'].unique()):
    subset = df_full[df_full['cluster'] == cluster]
    means = subset[FEATURES].mean()

    print(f"\nCluster {cluster}")
    print("-" * 40)

    print(f"Size: {len(subset)} users")
    print(f"Creator ratio: {means['creator_ratio']:.2f}")
    print(f"Interaction ratio: {means['interaction_ratio']:.2f}")
    print(f"Social ratio: {means['social_ratio']:.2f}")
    print(f"Reel preference: {means['reel_preference']:.2f}")

    # Behavior-based labeling
    if means['creator_ratio'] > global_means['creator_ratio'] * 1.5:
        name = "Content Creator"

    elif means['reel_preference'] > 0.6:
        name = "Reel Addict"

    elif means['social_ratio'] > global_means['social_ratio']:
        name = "Social User"

    else:
        name = "Passive Scroller"

    cluster_names[cluster] = name
    print(f"User Type: {name}")

# Assign labels
df_full['user_type'] = df_full['cluster'].map(cluster_names)

# ============================================================
# SAVE OUTPUT
# ============================================================

output_file = os.path.join(OUTPUT_PATH, "behavior_clustered_data.csv")
df_full.to_csv(output_file, index=False)

print(f"\nSaved clustered data at: {output_file}")

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_user(user_data):
    scaled = scaler.transform([user_data])
    cluster = kmeans.predict(scaled)[0]
    return cluster, cluster_names[cluster]

# Example prediction
example_user = X_full.iloc[0].values
cluster_id, user_type = predict_user(example_user)

print("\nExample Prediction:")
print("Cluster:", cluster_id)
print("User Type:", user_type)

print("\n✅ BEHAVIOR-BASED CLUSTERING COMPLETE")