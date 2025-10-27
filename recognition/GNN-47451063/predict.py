# Import Libraries
from train import run_model
from pathlib import Path

# Configure file paths
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "GNN-47451063/data"
edges_csv = data_dir / "musae_facebook_edges.csv"
target_csv = data_dir / "musae_facebook_target.csv"
feats_json = data_dir / "musae_facebook_features.json"
BASE_FOLDER = "GNN-47451063"

# Run the model
run_model(edges_path=edges_csv, target_path=target_csv, features_path=feats_json, base_folder=BASE_FOLDER)