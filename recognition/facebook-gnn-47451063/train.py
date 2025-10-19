import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import umap.umap_ as umap
from dataset import dataloader
from modules import GCNModel, GATModelBasic, GraphSAGE

def idx_accuracy(logits, y, idx):
    pred = logits.argmax(dim=-1)[idx]
    return accuracy_score(y[idx].cpu(), pred.cpu())

def train_and_eval(model, data, train_idx, valid_idx, test_idx, *,
                   lr=0.01, wd=5e-4, epochs=300, step_size=50, gamma=0.5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data = model.to(device), data.to(device)
    train_idx, valid_idx, test_idx = train_idx.to(device), valid_idx.to(device), test_idx.to(device)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = StepLR(opt, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    history = {
    "train_loss": [], "val_loss": [],
    "train_acc":  [], "val_acc":  [],
    "lr": []
    }

    best_val, best_state = -1.0, None
    patience, left = 80, 80

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data)
        
        tr_mask = data.y[train_idx] >= 0
        loss = criterion(out[train_idx][tr_mask], data.y[train_idx][tr_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)

            tr_logits = logits[train_idx][tr_mask]
            tr_labels = data.y[train_idx][tr_mask]
            train_pred = tr_logits.argmax(dim=-1)
            train_acc = (train_pred == tr_labels).float().mean().item()
            train_loss = loss.item()

            va_mask = (data.y[valid_idx] >= 0)
            if va_mask.sum() > 0:
                va_logits = logits[valid_idx][va_mask]
                va_labels = data.y[valid_idx][va_mask]
                val_loss = criterion(va_logits, va_labels).item()
                val_acc  = (va_logits.argmax(dim=-1) == va_labels).float().mean().item()
            else:
                val_loss, val_acc = float("nan"), 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(opt.param_groups[0]["lr"])

        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            left = patience
        else:
            left -= 1
        if left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(data)
        te_mask = data.y[test_idx] >= 0
        test_acc = idx_accuracy(logits, data.y, test_idx[te_mask]) if te_mask.sum() > 0 else float('nan')
    
    return test_acc, model, history

def show_umap(model_name, model, data, max_points=8000, seed=42, UMAP_N_NEIGHBORS=15, UMAP_MIN_DIST=0.05, UMAP_METRIC="cosine"):
    model.eval()
    with torch.no_grad():
        emb = model.embed(data).detach().cpu().numpy()
        labels = data.y.detach().cpu().numpy()

    # subsample for speed
    N = emb.shape[0]
    if max_points and N > max_points:
        idx = np.random.default_rng(seed).choice(N, size=max_points, replace=False)
        emb = emb[idx]; labels = labels[idx]

    keep = labels >= 0
    emb = emb[keep]; labels = labels[keep]

    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                        metric=UMAP_METRIC, random_state=seed)
    z = reducer.fit_transform(emb)
    plt.figure(figsize=(6,5))
    plt.scatter(z[:,0], z[:,1], c=labels, s=3, cmap="tab10")
    plt.title(f"UMAP — {model_name} embeddings")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(model_name + "_UMAP.png", dpi=200)

def plot_training_curves(model_name, history):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7,4.5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} — Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_name + "_LOSS_TRAINING_CURVE.png", dpi=200)

    plt.figure(figsize=(7,4.5))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"],   label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} — Accuracy")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_name + "_ACCURACY_TRAINING_CURVE.png", dpi=200)

def run_model(edges_path, 
              target_path, 
              feats_path, 
              svd_components=256, 
              seed=42, 
              MAX_UMAP=8000,
              UMAP_N_NEIGHBORS=15,
              UMAP_MIN_DIST=0.05,
              UMAP_METRIC="cosine"
              ):
    
    torch.manual_seed(seed)
    data, train_idx, valid_idx, test_idx, num_classes = dataloader(edges_path, target_path, feats_path, svd_components=svd_components, seed=seed)

    in_dim = data.x.size(1)
    out_dim = num_classes

    gcn_acc, gcn_model, gcn_hist  = train_and_eval(GCNModel(in_dim, 64, out_dim, dropout=0.6), data, train_idx, valid_idx, test_idx)

    print(f"GCN Test Accuracy: {gcn_acc:.4f}")
    show_umap("GCN", gcn_model, data, max_points=MAX_UMAP, UMAP_N_NEIGHBORS=UMAP_N_NEIGHBORS, UMAP_MIN_DIST=UMAP_MIN_DIST, UMAP_METRIC=UMAP_METRIC, seed=seed)
    plot_training_curves("GCN", gcn_hist)
