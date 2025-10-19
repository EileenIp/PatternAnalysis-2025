import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

def dataloader(edges_path, target_path, features_path, seed=42):
    edges_df = pd.read_csv(edges_path)
    targets_df = pd.read_csv(target_path)
    _ = features_path

    ids_from_edges = pd.unique(pd.concat([edges_df.iloc[:, 0], edges_df.iloc[:, 1]], axis=0))
    ids_from_targets = targets_df.iloc[:, 0].unique()
    all_ids = sorted(set(ids_from_edges) | set(ids_from_targets))
    id2idx = {int(nid): i for i, nid in enumerate(all_ids)}
    N = len(id2idx)

    source = edges_df.iloc[:, 0].astype(np.int64).map(id2idx).to_numpy(np.int64, copy=False)
    distance = edges_df.iloc[:, 1].astype(np.int64).map(id2idx).to_numpy(np.int64, copy=False)
    edge_index = torch.from_numpy(np.vstack((source, distance)))

    nid_col = targets_df.columns[0]
    label_col = None
    for c in targets_df.columns[1:]:
        if c.lower() in {"target", "label", "category", "page_type"}:
            label_col = c
            break
    if label_col is None and len(targets_df.columns) > 1:
        label_col = targets_df.columns[1]
    elif label_col is None:
        label_col = nid_col 

    ids = targets_df[nid_col].astype(np.int64).map(id2idx)

    if label_col == nid_col:
        y = torch.full((N,), -1, dtype=torch.long)
    else:
        raw_y = targets_df[label_col]
        if raw_y.dtype == object:
            y_vals, _ = pd.factorize(raw_y)
            y_series = pd.Series(y_vals, index=targets_df.index)
        else:
            y_series = raw_y.astype(np.int64)

        y = torch.full((N,), -1, dtype=torch.long)
        mask = ids.notna()
        y[torch.as_tensor(ids[mask].to_numpy(), dtype=torch.long)] = torch.as_tensor(
            y_series[mask].to_numpy(), dtype=torch.long
        )

    X = torch.eye(N, dtype=torch.float32)
    data = Data(x=X, edge_index=edge_index, y=y)
    
    return data
