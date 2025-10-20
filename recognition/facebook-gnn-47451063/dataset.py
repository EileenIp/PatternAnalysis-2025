# Import Libraries
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize
from torch_geometric.data import Data
from torch import Tensor
from typing import Dict, Iterable, List, Tuple


def load_inputs(
        edges_path: str, targets_path: str, 
        features_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[int]]]:
    edges_dataframe = pd.read_csv(edges_path)
    targets_dataframe = pd.read_csv(targets_path)
    with open(features_path, "r", encoding="utf-8") as f:
        features_map = json.load(f)
    return edges_dataframe, targets_dataframe, features_map

def collect_all_node_ids(
        edges_dataframe: pd.DataFrame, targets_dataframe: pd.DataFrame,
        features_map: Dict[str, Iterable[int]]) -> Tuple[List[int], Dict[int, int]]:
    # Extract node IDs from edges, targets, and features
    node_ids_from_edges = pd.unique(
        pd.concat([edges_dataframe.iloc[:, 0], edges_dataframe.iloc[:, 1]], axis=0)
    )
    node_ids_from_targets = targets_dataframe.iloc[:, 0].unique()
    node_ids_from_features = pd.Index([int(k) for k in features_map.keys()])

    # Combine and sort all unique node IDs
    all_node_ids = sorted(
        set(node_ids_from_edges) | set(node_ids_from_targets) | set(node_ids_from_features)
    )
    node_id_to_index = {int(node_id): i for i, node_id in enumerate(all_node_ids)}
    
    return all_node_ids, node_id_to_index

def build_edge_index(edges_dataframe: pd.DataFrame, node_id_to_index: Dict[int, int]) -> Tensor:
    # Map source and destination node ids to indices
    source_node_ids = (
        edges_dataframe.iloc[:, 0].astype(np.int64).map(node_id_to_index).to_numpy(np.int64, copy=False)
    )
    destination_node_ids = (
        edges_dataframe.iloc[:, 1].astype(np.int64).map(node_id_to_index).to_numpy(np.int64, copy=False)
    )

    # Build directed edge index
    directed_edges = torch.from_numpy(np.vstack((source_node_ids, destination_node_ids)))

    # Make undirected by adding the flipped edges [distance, source]
    return torch.cat([directed_edges, directed_edges.flip(0)], dim=1)

def find_label_column(targets_dataframe: pd.DataFrame) -> str:
    for column_name in targets_dataframe.columns[1:]:
        if column_name.lower() in {"target", "label", "category", "page_type"}:
            return column_name
    return targets_dataframe.columns[1]

def build_labels(targets_dataframe: pd.DataFrame, node_id_to_index: Dict[int, int], num_nodes: int) -> Tuple[Tensor, int]:
    node_id_column = targets_dataframe.columns[0]
    label_column = find_label_column(targets_dataframe)

    # Map node IDs to indices
    mapped_target_indices = targets_dataframe[node_id_column].astype(np.int64).map(node_id_to_index)

    # Convert labels to integers if they are strings; otherwise ensure int dtype
    raw_labels = targets_dataframe[label_column]
    if raw_labels.dtype == object:
        label_values, _ = pd.factorize(raw_labels)
        label_series = pd.Series(label_values, index=targets_dataframe.index)
    else:
        label_series = raw_labels.astype(np.int64)

    # Initialise all labels to -1 then fill where we have targets
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    present_mask = mapped_target_indices.notna()
    y[
        torch.as_tensor(mapped_target_indices[present_mask].to_numpy(), dtype=torch.long)
    ] = torch.as_tensor(label_series[present_mask].to_numpy(), dtype=torch.long)

    # Compute number of classes from labeled entries (handle all -1 edge case)
    num_classes = int(y[y >= 0].max().item() + 1) if (y >= 0).any() else 0
    
    return y, num_classes

def build_count_matrix(
        features_map: Dict[str, Iterable[int]], node_id_to_index: Dict[int, int], 
        num_nodes: int) -> sp.csr_matrix:
    row_indices: List[int] = []
    col_indices: List[int] = []
    values: List[float] = []

    # Aggregate counts per (node, feature) using Counter for robustness
    for node_id_str, feature_indices in features_map.items():
        node_index = node_id_to_index.get(int(node_id_str))
        if node_index is None or not feature_indices:
            continue
        counts = Counter(int(idx) for idx in feature_indices)
        for feature_index, term_frequency in counts.items():
            row_indices.append(node_index)
            col_indices.append(feature_index)
            values.append(float(term_frequency))

    # If there are no features, return an empty matrix with 0 columns
    if not row_indices:
        return sp.csr_matrix((num_nodes, 0), dtype=np.float32)

    # Assemble COO then convert to CSR for efficient arithmetic
    row_indices = np.asarray(row_indices, np.int64)
    col_indices = np.asarray(col_indices, np.int64)
    values = np.asarray(values, np.float32)
    feature_dimension = int(col_indices.max()) + 1
    
    #  Build COO matrix and return in CSR format
    return sp.coo_matrix((values, (row_indices, col_indices)), 
                         shape=(num_nodes, feature_dimension), dtype=np.float32).tocsr()


def apply_weights(count_matrix: sp.csr_matrix) -> sp.csr_matrix: 
    num_nodes = count_matrix.shape[0]
    document_frequencies = (count_matrix > 0).sum(axis=0).A1
    inverse_document_frequencies = np.log((1 + num_nodes) / (1 + document_frequencies)) + 1.0
    tfidf_matrix = count_matrix.multiply(inverse_document_frequencies)
    return sk_normalize(tfidf_matrix, norm="l2", axis=1, copy=False)


def svd_reduce(tfidf_matrix: sp.csr_matrix, svd_components: int, seed: int) -> np.ndarray:
    n_components = min(svd_components, max(2, tfidf_matrix.shape[1] - 1))
    svd_model = TruncatedSVD(n_components=n_components, random_state=seed)
    reduced_features = svd_model.fit_transform(tfidf_matrix)
    return sk_normalize(reduced_features, norm="l2", axis=1)

def build_features_tensor(
        features_map: Dict[str, Iterable[int]], node_id_to_index: Dict[int, int], num_nodes: int, svd_components: int, 
        seed: int) -> Tensor:
    count_matrix = build_count_matrix(features_map, node_id_to_index, num_nodes)

    # Handle edge case where there are no features
    if count_matrix.shape[1] == 0:
        return torch.zeros((num_nodes, svd_components), dtype=torch.float32)
    
    # Apply TF-IDF weighting and SVD reduction
    tfidf_matrix = apply_weights(count_matrix)
    reduced_features = svd_reduce(tfidf_matrix, svd_components, seed)

    return torch.from_numpy(reduced_features.astype(np.float32))


def random_splits(num_nodes: int, train_fraction: float = 0.8, val_fraction: float = 0.1) -> Tuple[Tensor, Tensor, Tensor]:

    # Generate a random permutation of node indices
    permutation_indices = torch.randperm(num_nodes)
    num_train = int(train_fraction * num_nodes)
    num_val = int(val_fraction * num_nodes)

    # Split indices into train, val, test
    train_idx = permutation_indices[:num_train]
    valid_idx = permutation_indices[num_train : num_train + num_val]
    test_idx = permutation_indices[num_train + num_val :]

    return train_idx, valid_idx, test_idx


def dataloader(
        edges_path: str, target_path: str, features_path: str, seed: int = 42, 
        svd_components: int = 256) -> Tuple[Data, Tensor, Tensor, Tensor, int]:
    # 1) Load raw inputs from disk
    edges_dataframe, targets_dataframe, features_map = load_inputs(edges_path, target_path, features_path)

    # 2) Build node universe and id->index mapping
    all_node_ids, node_id_to_index = collect_all_node_ids(edges_dataframe, targets_dataframe, features_map)
    num_nodes = len(all_node_ids)

    # 3) Construct graph connectivity (undirected)
    edge_index = build_edge_index(edges_dataframe, node_id_to_index)

    # 4) Build labels vector y and count classes
    y, num_classes = build_labels(targets_dataframe, node_id_to_index, num_nodes)

    # 5) Build node feature matrix x via TF-IDF + SVD
    x = build_features_tensor(features_map, node_id_to_index, num_nodes, svd_components, seed)

    # 6) Package into a PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    # 7) Create random train/val/test splits
    train_idx, valid_idx, test_idx = random_splits(data.num_nodes, train_fraction=0.8, val_fraction=0.1)

    return data, train_idx, valid_idx, test_idx, num_classes