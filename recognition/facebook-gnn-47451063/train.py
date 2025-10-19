import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from dataset import dataloader
from modules import GCNModel

def train_and_eval(model, data, train_idx, valid_idx, test_idx, *,
                   lr=0.01, wd=5e-4, epochs=300, step_size=50, gamma=0.5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data = model.to(device), data.to(device)
    train_idx, valid_idx, test_idx = train_idx.to(device), valid_idx.to(device), test_idx.to(device)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = StepLR(opt, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    patience, left = 80, 80

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data)
        
        # use only labeled nodes in the splits
        tr_mask = data.y[train_idx] >= 0
        loss = criterion(out[train_idx][tr_mask], data.y[train_idx][tr_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
            va_mask = (data.y[valid_idx] >= 0)
            if va_mask.sum() > 0:
                va_logits = logits[valid_idx][va_mask]
                va_labels = data.y[valid_idx][va_mask]
                val_acc  = (va_logits.argmax(dim=-1) == va_labels).float().mean().item()

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
        pred = logits.argmax(dim=-1)[test_idx[te_mask]]
        test_acc = accuracy_score(data.y[test_idx[te_mask]].cpu(), pred.cpu())
        if te_mask.sum() > 0:
            test_acc = test_acc
        else:
            test_acc = float('nan') 
    
    return test_acc

def run_model(edges_path, 
              target_path, 
              feats_path, 
              seed=42, 
              ):
    
    torch.manual_seed(seed)
    data, train_idx, valid_idx, test_idx, num_classes = dataloader(edges_path, target_path, feats_path, seed=seed)

    in_dim = data.x.size(1)
    out_dim = num_classes

    gcn_accuracy = train_and_eval(GCNModel(in_dim, 64, out_dim, dropout=0.6), data, train_idx, valid_idx, test_idx)

    print(f"GCN Test Accuracy: {gcn_accuracy:.4f}")
