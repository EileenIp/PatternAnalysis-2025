# Import Libraries
from train import run_model
from pathlib import Path

# Configure file paths
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "GNN-47451063/data"
EDGES = data_dir / "musae_facebook_edges.csv"
TARGET = data_dir / "musae_facebook_target.csv"
FEATURES = data_dir / "musae_facebook_features.json"
BASE_FOLDER = "GNN-47451063"

# Configure model parameters
MODEL_PARAMETERS = {
    "GCN": {
        "input_dim": None,
        "output_dim": None,
        "hidden_dim": 64,
        "dropout": 0.6,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "epochs": 300,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.5,
        "grad_clip_max_norm": 2.0,
        "patience": 80,
    },
    "GAT": {
        "input_dim": None,
        "output_dim": None,
        "hidden_dim": 64,
        "dropout": 0.6,
        "heads": 8,
        "learning_rate": 0.005,
        "weight_decay": 5e-4,
        "epochs": 300,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.5,
        "grad_clip_max_norm": 2.0,
        "patience": 80,
    },
    "SAGE": {
        "input_dim": None,
        "output_dim": None,
        "hidden_dim": 64,
        "dropout": 0.6,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "epochs": 300,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.5,
        "grad_clip_max_norm": 2.0,
        "patience": 80,
    },
}

# Run the model
run_model(edges_path=EDGES, target_path=TARGET, features_path=FEATURES,
           base_folder=BASE_FOLDER, model_parameters=MODEL_PARAMETERS)