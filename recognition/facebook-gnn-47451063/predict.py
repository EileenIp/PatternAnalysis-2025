from train import run_model
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"
edges_csv = data_dir / "musae_facebook_edges.csv"
target_csv = data_dir / "musae_facebook_target.csv"
feats_json = data_dir / "musae_facebook_features.json"

SEED = 42
SVD_COMPONENTS = 256

MAX_UMAP = 8000
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "cosine"


run_model(edges_csv, target_csv, feats_json, 
          svd_components=SVD_COMPONENTS, 
          seed=SEED, 
          MAX_UMAP=MAX_UMAP,
          UMAP_N_NEIGHBORS=UMAP_N_NEIGHBORS,
          UMAP_MIN_DIST=UMAP_MIN_DIST,
          UMAP_METRIC=UMAP_METRIC
          )