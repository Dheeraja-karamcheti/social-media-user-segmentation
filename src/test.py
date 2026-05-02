"""
Behavior-Based Social Media User Segmentation (Final Version)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# PATH SETUP
# ============================================================

USE_LOCAL_PATH = True

if USE_LOCAL_PATH:
    PROJECT_ROOT = r"D:\social_media_clustering_project"
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "instagram_usage_lifestyle.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()
print("Dataset loaded:", df.shape)

# ============================================================
# CLEANING
# ============================================================

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

epsilon = 1e-6

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

FEATURES = [
    'age',
    'creator_ratio',
    'interaction_ratio',
    'social_ratio',
    'reel_preference',
    'scroll_ratio',
    'avg_session_intensity'
]

print(f"\nUsing {len(FEATURES)} features")

df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

# ============================================================
# OUTLIER HANDLING
# ============================================================

for col in FEATURES:
    df[col] = np.clip(
        df[col],
        df[col].quantile(0.01),
        df[col].quantile(0.99)
    )

# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])

# ============================================================
# ELBOW METHOD
# ============================================================

print("\nRunning Elbow Method...")

sample_size = 50000
indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[indices]

inertia = []
K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_sample)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)

elbow_path = os.path.join(OUTPUT_PATH, "elbow_plot.png")
plt.savefig(elbow_path)
plt.close()

print(f"Elbow plot saved at: {elbow_path}")

# ============================================================
# FINAL MODEL
# ============================================================

K = 4  # adjust based on elbow

print(f"\nTraining final model with K = {K}")

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster distribution:")
print(df['cluster'].value_counts())

# ============================================================
# PCA VISUALIZATION
# ============================================================

print("\nGenerating PCA visualization...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], alpha=0.5)
plt.title("User Clusters (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

pca_path = os.path.join(OUTPUT_PATH, "clusters_pca.png")
plt.savefig(pca_path)
plt.close()

print(f"PCA plot saved at: {pca_path}")

# ============================================================
# CLUSTER INTERPRETATION (FIXED LOGIC)
# ============================================================

print("\nCluster Profiles:")

cluster_summary = {}

for cluster in sorted(df['cluster'].unique()):
    subset = df[df['cluster'] == cluster]
    means = subset[FEATURES].mean()

    cluster_summary[cluster] = {
        'creator_ratio': means['creator_ratio'],
        'reel_preference': means['reel_preference'],
        'social_ratio': means['social_ratio'],
        'interaction_ratio': means['interaction_ratio']
    }

summary_df = pd.DataFrame(cluster_summary).T
print("\nCluster Feature Summary:\n", summary_df)

cluster_names = {}

for cluster in summary_df.index:
    c = summary_df.loc[cluster]

    if c['creator_ratio'] == summary_df['creator_ratio'].max():
        name = "Top Content Creators"

    elif c['reel_preference'] == summary_df['reel_preference'].max():
        name = "Reel-Focused Users"

    elif c['social_ratio'] == summary_df['social_ratio'].max():
        name = "Highly Social Users"

    else:
        if c['interaction_ratio'] > summary_df['interaction_ratio'].median():
            name = "Active Engagers"
        else:
            name = "Low Engagement Users"

    cluster_names[cluster] = name

    print(f"\nCluster {cluster}")
    print("----------------------------------------")
    print(f"Size: {len(df[df['cluster'] == cluster])}")
    print(f"User Type: {name}")

df['user_type'] = df['cluster'].map(cluster_names)

# ============================================================
# SAVE OUTPUT
# ============================================================

output_file = os.path.join(OUTPUT_PATH, "clustered_data.csv")
df.to_csv(output_file, index=False)

print(f"\nSaved clustered data at: {output_file}")

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_user(user_dict):
    input_df = pd.DataFrame([user_dict])
    input_df = input_df[FEATURES]
    scaled = scaler.transform(input_df)
    cluster = kmeans.predict(scaled)[0]
    return cluster, cluster_names[cluster]

# Example prediction
example_user = df[FEATURES].iloc[0].to_dict()
cluster_id, user_type = predict_user(example_user)

print("\nExample Prediction:")
print("Cluster:", cluster_id)
print("User Type:", user_type)

print("\n✅ FINAL CLUSTERING COMPLETE")