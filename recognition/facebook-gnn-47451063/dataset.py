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


def load_data_files(
        edges_path: str, targets_path: str, 
        features_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[int]]]:
    """
    Load the input data from the specified file paths.
    
    Args:
        edges_path (str): Path to the CSV file containing edge data.
        targets_path (str): Path to the CSV file containing target labels.
        features_path (str): Path to the JSON file containing node features.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[int]]]: DataFrames for edges and targets, and a dictionary for features.
    """
    edges_dataframe = pd.read_csv(edges_path)
    targets_dataframe = pd.read_csv(targets_path)
    with open(features_path, "r", encoding="utf-8") as f:
        features_map = json.load(f)
    return edges_dataframe, targets_dataframe, features_map

def collect_node_ids(
        edges_dataframe: pd.DataFrame, targets_dataframe: pd.DataFrame,
        features_map: Dict[str, Iterable[int]]) -> Tuple[List[int], Dict[int, int]]:
    """
    Collect all unique node IDs from edges, targets, and features, and create a mapping to adjacent indices.

    Args:
        edges_dataframe (pd.DataFrame): DataFrame containing edge data.
        targets_dataframe (pd.DataFrame): DataFrame containing target labels.
        features_map (Dict[str, Iterable[int]]): Dictionary mapping node IDs to their feature indices.

    Returns:
        Tuple[List[int], Dict[int, int]]: A sorted list of all unique node IDs and a mapping from node ID to adjacent index.
    """
    # Extract node IDs from edges, targets, and features
    node_ids_from_edges = pd.unique(pd.concat([edges_dataframe.iloc[:, 0], edges_dataframe.iloc[:, 1]], axis=0))
    node_ids_from_targets = targets_dataframe.iloc[:, 0].unique()
    node_ids_from_features = pd.Index([int(k) for k in features_map.keys()])

    # Combine and sort all unique node IDs
    all_node_ids = sorted(set(node_ids_from_edges) | set(node_ids_from_targets) | set(node_ids_from_features))
    node_id_to_index = {int(node_id): i for i, node_id in enumerate(all_node_ids)}
    
    return all_node_ids, node_id_to_index

def build_edge_index(edges_dataframe: pd.DataFrame, node_id_to_index: Dict[int, int]) -> Tensor:
    """
    Build the edge index tensor from the edges and node ID to index mapping.

    Args:
        edges_dataframe (pd.DataFrame): DataFrame containing edge data.
        node_id_to_index (Dict[int, int]): Dictionary mapping from node ID to adjacent index.

    Returns:
        Tesnor: Edge index tensor.
    """
    # Map node IDs to indices
    source_node_ids = (edges_dataframe.iloc[:, 0].astype(np.int64).map(node_id_to_index).to_numpy(np.int64, copy=False))
    destination_node_ids = (edges_dataframe.iloc[:, 1].astype(np.int64).map(node_id_to_index).to_numpy(np.int64, copy=False))

    # Build directed edge index
    directed_edges = torch.from_numpy(np.vstack((source_node_ids, destination_node_ids)))

    # Make undirected by adding flipped edges
    return torch.cat([directed_edges, directed_edges.flip(0)], dim=1)

def find_label_column(targets_dataframe: pd.DataFrame) -> str:
    """
    Identify the label column in the targets dataset.

    Args:
        targets_dataframe (pd.DataFrame): DataFrame containing target labels.

    Returns:
        str: Name of the label column.
    """
    for column_name in targets_dataframe.columns[1:]:
        if column_name.lower() in {"target", "label", "category", "page_type"}:
            return column_name
    return targets_dataframe.columns[1]

def build_labels(targets_dataframe: pd.DataFrame, node_id_to_index: Dict[int, int], num_nodes: int) -> Tuple[Tensor, int]:
    """
    Build the labels tensor and count the number of classes.

    Args:
        targets_dataframe (pd.DataFrame): DataFrame containing target labels.
        node_id_to_index (Dict[int, int]): Dictionary mapping from node ID to adjacent index.
        num_nodes (int): Total number of nodes.

    Returns:
        Tuple[Tensor, int]: Labels tensor and number of classes.
    """
    node_id_column = targets_dataframe.columns[0]
    label_column = find_label_column(targets_dataframe)

    # Map node IDs to indices
    mapped_target_indices = targets_dataframe[node_id_column].astype(np.int64).map(node_id_to_index)

    # Convert labels to integers if they are strings, otherwise ensure int datatype
    labels = targets_dataframe[label_column]
    if labels.dtype == object:
        label_values, _ = pd.factorize(labels)
        label_series = pd.Series(label_values, index=targets_dataframe.index)
    else:
        label_series = labels.astype(np.int64)

    # Build labels tensor with -1 for unlabeled nodes
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    present_mask = mapped_target_indices.notna()
    y[torch.as_tensor(mapped_target_indices[present_mask].to_numpy(), dtype=torch.long)] = torch.as_tensor(label_series[present_mask].to_numpy(), dtype=torch.long)

    # Count number of classes
    num_classes = int(y[y >= 0].max().item() + 1) if (y >= 0).any() else 0
    
    return y, num_classes

def build_matrix(
        features_map: Dict[str, Iterable[int]], node_id_to_index: Dict[int, int], 
        num_nodes: int) -> sp.csr_matrix:
    """
    Build a sparse count matrix from the features map.

    Args:
        features_map (Dict[str, Iterable[int]]): Dictionary mapping node IDs to their feature indices.
        node_id_to_index (Dict[int, int]): Dictionary mapping from node ID to contiguous index.
        num_nodes (int): Total number of nodes.

    Returns:
        sp.csr_matrix: Sparse count matrix of shape (num_nodes, feature_dimension).
    """
    row_indices: List[int] = []
    col_indices: List[int] = []
    values: List[float] = []

    # Create sparse matrix entries
    for node_id_str, feature_indices in features_map.items():
        node_index = node_id_to_index.get(int(node_id_str))
        if node_index is None or not feature_indices:
            continue
        counts = Counter(int(idx) for idx in feature_indices)
        for feature_index, term_frequency in counts.items():
            row_indices.append(node_index)
            col_indices.append(feature_index)
            values.append(float(term_frequency))

    # If there are no features, return an empty matrix
    if not row_indices:
        return sp.csr_matrix((num_nodes, 0), dtype=np.float32)

    # Convert to numpy arrays
    row_indices = np.asarray(row_indices, np.int64)
    col_indices = np.asarray(col_indices, np.int64)
    values = np.asarray(values, np.float32)
    feature_dimension = int(col_indices.max()) + 1
    
    # Build and return the sparse matrix
    return sp.coo_matrix((values, (row_indices, col_indices)), 
                         shape=(num_nodes, feature_dimension), dtype=np.float32).tocsr()


def apply_weights(count_matrix: sp.csr_matrix) -> sp.csr_matrix:
    """
    Compute TF-IDF weighting and L2-normalise rows of the count matrix.

    Args:
        count_matrix (sp.csr_matrix): Sparse count matrix.

    Returns:
        sp.csr_matrix: TF-IDF weighted and L2-normalised sparse matrix.
    """
    num_nodes = count_matrix.shape[0]
    
    # Compute TF-IDF weights
    document_frequencies = (count_matrix > 0).sum(axis=0).A1
    inverse_document_frequencies = np.log((1 + num_nodes) / (1 + document_frequencies)) + 1.0
    tfidf_matrix = count_matrix.multiply(inverse_document_frequencies)

    # Normalise the TF-IDF matrix
    return sk_normalize(tfidf_matrix, norm="l2", axis=1, copy=False)


def reduce_features(tfidf_matrix: sp.csr_matrix, svd_components: int, seed: int) -> np.ndarray:
    """
    Reduce dimensionality of TF-IDF features using Truncated SVD and normalise.

    Args:
        tfidf_matrix (sp.csr_matrix): Sparse matrix.
        svd_components (int): Number of SVD components for reduction.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Reduced and normalised feature matrix.
    """
    n_components = min(svd_components, max(2, tfidf_matrix.shape[1] - 1))

    # Apply Truncated SVD
    svd_model = TruncatedSVD(n_components=n_components, random_state=seed)
    reduced_features = svd_model.fit_transform(tfidf_matrix)

    # Normalise the reduced features
    return sk_normalize(reduced_features, norm="l2", axis=1)

def build_features_tensor(
        features_map: Dict[str, Iterable[int]], node_id_to_index: Dict[int, int], num_nodes: int, svd_components: int, 
        seed: int) -> Tensor:
    """
    Build the node features tensor via TF-IDF weighting and SVD reduction.

    Args:
        features_map (Dict[str, Iterable[int]]): Dictionary mapping node IDs to their feature indices.
        node_id_to_index (Dict[int, int]): Dictionary mapping from node ID to contiguous index.
        num_nodes (int): Total number of nodes.
        svd_components (int): Number of SVD components for feature reduction.
        seed (int): Random seed for reproducibility.

    Returns:
        Tensor: Node features tensor.
    """
    # Build the sparse count matrix
    count_matrix = build_matrix(features_map, node_id_to_index, num_nodes)

    # Handle edge case where there are no features
    if count_matrix.shape[1] == 0:
        return torch.zeros((num_nodes, svd_components), dtype=torch.float32)
    
    # Apply TF-IDF weighting and reduce dimensionality
    tfidf_matrix = apply_weights(count_matrix)
    reduced_features = reduce_features(tfidf_matrix, svd_components, seed)

    return torch.from_numpy(reduced_features.astype(np.float32))


def split_data(num_nodes: int, train_size: float = 0.8, val_size: float = 0.1) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Split the nodes into training, validation, and test sets.

    Args:
        num_nodes (int): Total number of nodes.
        train_size (float): Size of nodes to use for training.
        val_size (float): Size of nodes to use for validation.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Tensors containing indices for training, validation, and test sets.
    """
    # Generate a random permutation of node indices
    permutation_indices = torch.randperm(num_nodes)
    num_train = int(train_size * num_nodes)
    num_val = int(val_size * num_nodes)

    # Split indices into train, val, test sets
    train_idx = permutation_indices[:num_train]
    valid_idx = permutation_indices[num_train : num_train + num_val]
    test_idx = permutation_indices[num_train + num_val :]

    return train_idx, valid_idx, test_idx


def dataloader(
        edges_path: str, target_path: str, features_path: str, seed: int = 42, 
        svd_components: int = 256) -> Tuple[Data, Tensor, Tensor, Tensor, int]:
    """
    Load and preprocess the dataset, returning a Data object and train/val/test splits.

    Args:
        edges_path (str): Path to the CSV file containing edge data.
        target_path (str): Path to the CSV file containing target labels.
        features_path (str): Path to the JSON file containing node features.
        seed (int): Random seed for reproducibility.
        svd_components (int): Number of SVD components for feature reduction.

    Returns:
        Tuple[Data, Tensor, Tensor, Tensor, int]: Data object, train, validation, test indices, and number of classes.
    """
    # Load data files
    edges_dataframe, targets_dataframe, features_map = load_data_files(edges_path, target_path, features_path)

    # Collect all unique node IDs and create universal mapping
    all_node_ids, node_id_to_index = collect_node_ids(edges_dataframe, targets_dataframe, features_map)
    num_nodes = len(all_node_ids)

    # Build edge index tensor
    edge_index = build_edge_index(edges_dataframe, node_id_to_index)

    # Build labels tensor and count number of classes
    y, num_classes = build_labels(targets_dataframe, node_id_to_index, num_nodes)

    # Build features tensor with TF-IDF weighting and SVD feature reduction
    x = build_features_tensor(features_map, node_id_to_index, num_nodes, svd_components, seed)

    # Create Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    # Create train/val/test splits
    train_idx, valid_idx, test_idx = split_data(data.num_nodes, train_size=0.8, val_size=0.1)

    return data, train_idx, valid_idx, test_idx, num_classes