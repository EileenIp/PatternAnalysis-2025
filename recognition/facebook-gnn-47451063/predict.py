# Import Libraries
from train import run_model
from pathlib import Path

# Define file paths
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"
edges_csv = data_dir / "musae_facebook_edges.csv"
target_csv = data_dir / "musae_facebook_target.csv"
feats_json = data_dir / "musae_facebook_features.json"

# Configuration parameters
SEED = 42
SVD_COMPONENTS = 256
MAX_TSNE = 8000
TSNE_PERPLEXITY = 30
TSNE_ITER = 1000
MAX_UMAP = 8000
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "cosine"
DEVICE = None

# Run the model
run_model(
    edges_path=edges_csv,
    target_path=target_csv,
    feats_path=feats_json,
    svd_components=SVD_COMPONENTS,
    seed=SEED,
    device=DEVICE,
    max_tsne_nodes=MAX_TSNE,
    tsne_perplexity=TSNE_PERPLEXITY,
    tsne_n_iter=TSNE_ITER,
    max_umap_nodes=MAX_UMAP,
    umap_n_neighbors=UMAP_N_NEIGHBORS,
    umap_min_dist=UMAP_MIN_DIST,
    umap_metric=UMAP_METRIC

)