# predict.py
from train import run_model
from pathlib import Path

# Configuration
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"
edges_csv = data_dir / "musae_facebook_edges.csv"
target_csv = data_dir / "musae_facebook_target.csv"
feats_json = data_dir / "musae_facebook_features.json"

SEED = 42
SVD_COMPONENTS = 256   # reduce BoW features to this dim for speed+memory

run_model(edges_csv, target_csv, feats_json, 
          svd_components=SVD_COMPONENTS, 
          seed=SEED, 
          )