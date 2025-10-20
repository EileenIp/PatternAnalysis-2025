# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import umap.umap_ as umap
from dataset import dataloader
from modules import GCN, GAT, SAGE
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from typing import Dict, List, Tuple

def train(
        model: nn.Module, data, train_indices, val_indices, learning_rate: float = 0.01, weight_decay: float = 5e-4,
        epochs: int = 300, scheduler_step_size: int = 50, scheduler_gamma: float = 0.5, device=None, 
        grad_clip_max_norm: float = 2.0, patience: int = 80) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the GNN model with early stopping and return the best model and training history.

    Args:
        model (nn.Module): The GNN model to train.
        data: The graph data.
        train_indices: Indices of training nodes.
        val_indices: Indices of validation nodes.
        learning_rate (float): Learning rate for the optimiser.
        weight_decay (float): Weight decay (L2 regularisation) for the optimiser.
        epochs (int): Maximum number of training epochs.
        scheduler_step_size (int): Step size for the learning rate scheduler.
        scheduler_gamma (float): Multiplicative factor for learning rate decay.
        device: Device to run the training on (CPU or GPU).
        grad_clip_max_norm (float): Maximum norm for gradient clipping.
        patience (int): Number of epochs to wait for improvement before early stopping.
    
    Returns:
        Tuple[nn.Module, Dict[str, List[float]]]: The best model and training history.
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, data = model.to(device), data.to(device)
    train_indices = train_indices.to(device)
    val_indices = val_indices.to(device)

    # Set optimiser, scheduler, loss
    optimiser = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimiser, step_size=scheduler_step_size, gamma=scheduler_gamma)
    loss_fn = nn.CrossEntropyLoss()

    # Track history and best model
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_accuracy = -1.0
    best_state_dict = None
    remaining_patience = patience

    # Training loop
    for _ in range(epochs):
        model.train()
        optimiser.zero_grad()
        logits = model(data)

        train_label_mask = data.y[train_indices] >= 0
        loss = loss_fn(logits[train_indices][train_label_mask], data.y[train_indices][train_label_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
        optimiser.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
            # Training metrics
            train_logits = logits[train_indices][train_label_mask]
            train_labels = data.y[train_indices][train_label_mask]
            train_accuracy = (train_logits.argmax(dim=-1) == train_labels).float().mean().item()
            train_loss = loss.item()
            # Validation metrics
            val_label_mask = (data.y[val_indices] >= 0)
            if val_label_mask.sum() > 0:
                val_logits = logits[val_indices][val_label_mask]
                val_labels = data.y[val_indices][val_label_mask]
                val_loss = loss_fn(val_logits, val_labels).item()
                val_accuracy = (val_logits.argmax(dim=-1) == val_labels).float().mean().item()
            else:
                val_loss, val_accuracy = float("nan"), 0.0

        # Record loss and accuracy metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(val_accuracy)
        history["lr"].append(optimiser.param_groups[0]["lr"])

        # Early stopping check
        if val_accuracy > best_val_accuracy + 1e-4:
            best_val_accuracy = val_accuracy
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            remaining_patience = patience
        else:
            remaining_patience -= 1

        if remaining_patience <= 0:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history


def evaluate(model: nn.Module, data, test_indices, device=None) -> float:
    """
    Evaluate the model on the test set and return accuracy.

    Args:
        model (nn.Module): The trained GNN model.
        data: The graph data.
        test_indices: Indices of test nodes.
        device: Device to run the evaluation on (CPU or GPU).
    
    Returns:
        float: Test accuracy.
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, data = model.to(device), data.to(device)
    test_indices = test_indices.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(data)
        test_label_mask = data.y[test_indices] >= 0
        
        if test_label_mask.sum() == 0:
            return float("nan")

        # Calculate accuracy
        predictions = logits.argmax(dim=-1)[test_indices[test_label_mask]]
        test_accuracy = accuracy_score(data.y[test_indices[test_label_mask]].cpu(), predictions.cpu())

    return test_accuracy


def build_tsne(
        model_name: str, model: nn.Module, data, max_points: int = 800, seed: int = 42, n_components: int = 2, 
        perplexity: float = 30, n_iter: int = 1000, init: str = "pca", learning_rate="auto") -> None:
    """
    Generate and save t-SNE plot of node embeddings.

    Args:
        model_name (str): Name of the model (for saving the plot).
        model (nn.Module): The trained GNN model.
        data: The graph data.
        max_points (int): Maximum number of points to plot.
        seed (int): Random seed for reproducibility.
        n_components (int): Number of t-SNE components.
        perplexity (float): t-SNE perplexity parameter.
        n_iter (int): Number of t-SNE iterations.
        init (str): Initialization method for t-SNE.
        learning_rate (str or float): Learning rate for t-SNE.
    """
    # Get embeddings and labels
    model.eval()
    with torch.no_grad():
        embeddings = model.embed(data).detach().cpu().numpy()
        labels = data.y.detach().cpu().numpy()

    # Subsample if necessary
    num_nodes = embeddings.shape[0]
    if max_points and num_nodes > max_points:
        sample_indices = np.random.default_rng(seed).choice(num_nodes, size=max_points, replace=False)
        embeddings = embeddings[sample_indices]
        labels = labels[sample_indices]

    # Filter out unlabeled nodes
    labeled_mask = labels >= 0
    embeddings = embeddings[labeled_mask]
    labels = labels[labeled_mask]

    # Bulid and fit t-SNE
    tsne = TSNE(n_components=n_components,
                random_state=seed,
                perplexity=perplexity,
                n_iter=n_iter,
                init=init,
                learning_rate=learning_rate
                )
    embedding_2d = tsne.fit_transform(embeddings)

    # Plot and save t-SNE
    plt.figure(figsize=(6, 5))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, s=3, cmap="tab10")
    plt.title(f"t-SNE — {model_name} embeddings")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.tight_layout()
    plt.savefig(model_name + "_TSNE.png", dpi=200)


def build_umap(
        model_name: str, model: nn.Module, data, max_points: int = 8000, seed: int = 42, umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.05, umap_metric: str = "cosine") -> None:
    # Get embeddings and labels
    model.eval()
    with torch.no_grad():
        embeddings = model.embed(data).detach().cpu().numpy()
        labels = data.y.detach().cpu().numpy()

    # Subsample if necessary
    num_nodes = embeddings.shape[0]
    if max_points and num_nodes > max_points:
        sample_indices = np.random.default_rng(seed).choice(num_nodes, size=max_points, replace=False)
        embeddings = embeddings[sample_indices]
        labels = labels[sample_indices]

    # Filter out unlabeled nodes
    labeled_mask = labels >= 0
    embeddings = embeddings[labeled_mask]
    labels = labels[labeled_mask]

    # Build and fit UMAP
    reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=umap_metric, random_state=seed)
    embedding_2d = reducer.fit_transform(embeddings)

    # Plot and save UMAP
    plt.figure(figsize=(6, 5))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, s=3, cmap="tab10")
    plt.title(f"UMAP — {model_name} embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(model_name + "_UMAP.png", dpi=200)


def build_training_curves(model_name: str, history: Dict[str, List[float]]):
    epochs_axis = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs_axis, history["train_loss"], label="Train Loss")
    plt.plot(epochs_axis, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} — Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_name + "_LOSS_TRAINING_CURVE.png", dpi=200)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs_axis, history["train_acc"], label="Train Acc")
    plt.plot(epochs_axis, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} — Accuracy")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_name + "_ACCURACY_TRAINING_CURVE.png", dpi=200)


def run_models(
        edges_path: str, target_path: str, features_path: str, svd_components: int = 256, seed: int = 42, device=None,
        max_umap_points: int = 8000, max_tsne_points: int = 8000, tsne_perplexity: int = 30, umap_n_neighbors: int = 15,
        tsne_iterations: int = 1000, umap_min_dist: float = 0.05, umap_metric: str = "cosine") -> None:
    torch.manual_seed(seed)

    # Load data
    data, train_indices, val_indices, test_indices, num_classes = dataloader(edges_path, target_path, features_path,
                                                                             svd_components=svd_components, seed=seed)

    input_dim = data.x.size(1)
    output_dim = num_classes

    # Define models to train
    model_configs = [
        ("GCN", lambda: GCN(input_dim, 64, output_dim, dropout=0.6), dict(learning_rate=0.01)),
        ("GAT", lambda: GAT(input_dim, 64, output_dim, dropout=0.6, heads=8), dict(learning_rate=0.005)),
        ("SAGE", lambda: SAGE(input_dim, 64, output_dim, dropout=0.6), dict(learning_rate=0.01)),
    ]

    results: Dict[str, float] = {}
    trained_models: Dict[str, nn.Module] = {}
    histories: Dict[str, Dict[str, List[float]]] = {}

    for model_name, model_factory, hyperparams in model_configs:
        # Train and get best model
        model = model_factory()
        model, history = train(model=model, data=data, train_indices=train_indices, val_indices=val_indices,
                               learning_rate=hyperparams.get("learning_rate", 0.01), weight_decay=5e-4, epochs=300,
                               scheduler_step_size=50, scheduler_gamma=0.5, device=device, patience=80)

        # Evaluate the model
        test_accuracy = evaluate(model, data, test_indices)
        results[model_name] = test_accuracy
        trained_models[model_name] = model
        histories[model_name] = history

        print(f"{model_name}: {test_accuracy}")

        # Visualise results
        build_tsne(model_name=model_name, model=model, data=data, max_points=max_tsne_points, perplexity=tsne_perplexity,
                  n_iter=tsne_iterations, seed=seed)
        build_umap(model_name=model_name, model=model, data=data, max_points=max_umap_points, 
                   umap_n_neighbors=umap_n_neighbors, umap_min_dist=umap_min_dist, umap_metric=umap_metric, seed=seed)
        build_training_curves(model_name, history)