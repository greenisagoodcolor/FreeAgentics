"""
Module for FreeAgentics Active Inference implementation.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from .batch_processor import GraphData

"""
Benchmark Datasets for GNN Testing
Provides standard graph datasets for benchmarking and validation.
"""


@dataclass
class DatasetInfo:
    """Information about a benchmark dataset"""

    name: str
    num_graphs: int
    num_classes: int
    avg_nodes: float
    avg_edges: float
    node_features: int
    edge_features: Optional[int] = None
    task_type: str = "graph_classification"
    description: str = ""


class BenchmarkDatasets:
    """
    Provides standard benchmark datasets for GNN testing.
    Includes synthetic and real-world inspired datasets for:
    - Node classification
    - Graph classification
    - Link prediction
    - Graph regression
    """

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize benchmark datasets.
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def get_dataset_info(self) -> Dict[str, DatasetInfo]:
        """Get information about available datasets"""
        return {
            "karate_club": DatasetInfo(
                name="Zachary Karate Club",
                num_graphs=1,
                num_classes=2,
                avg_nodes=34,
                avg_edges=78,
                node_features=34,
                task_type="node_classification",
                description="Classic social network dataset",
            ),
            "synthetic_small": DatasetInfo(
                name="Synthetic Small Graphs",
                num_graphs=100,
                num_classes=3,
                avg_nodes=15,
                avg_edges=30,
                node_features=32,
                task_type="graph_classification",
                description="Small synthetic graphs for quick testing",
            ),
            "synthetic_medium": DatasetInfo(
                name="Synthetic Medium Graphs",
                num_graphs=100,
                num_classes=5,
                avg_nodes=50,
                avg_edges=150,
                node_features=64,
                edge_features=8,
                task_type="graph_classification",
                description="Medium-sized synthetic graphs",
            ),
            "synthetic_large": DatasetInfo(
                name="Synthetic Large Graphs",
                num_graphs=50,
                num_classes=10,
                avg_nodes=200,
                avg_edges=800,
                node_features=128,
                edge_features=16,
                task_type="graph_classification",
                description="Large synthetic graphs for stress testing",
            ),
            "grid_graphs": DatasetInfo(
                name="Grid Graphs",
                num_graphs=100,
                num_classes=4,
                avg_nodes=25,
                avg_edges=40,
                node_features=16,
                task_type="graph_classification",
                description="2D grid-based graphs",
            ),
            "tree_graphs": DatasetInfo(
                name="Tree Graphs",
                num_graphs=100,
                num_classes=3,
                avg_nodes=31,
                avg_edges=30,
                node_features=32,
                task_type="graph_classification",
                description="Tree-structured graphs",
            ),
            "community_graphs": DatasetInfo(
                name="Community Graphs",
                num_graphs=50,
                num_classes=4,
                avg_nodes=100,
                avg_edges=500,
                node_features=64,
                task_type="node_classification",
                description="Graphs with community structure",
            ),
        }

    def load_karate_club(self) -> List[GraphData]:
        """
        Load Zachary's Karate Club dataset.
        Returns:
            List containing single graph
        """
        G = nx.karate_club_graph()
        num_nodes = G.number_of_nodes()
        node_features = torch.eye(num_nodes, dtype=torch.float32)
        edge_list = list(G.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        labels = torch.tensor(
            [G.nodes[i]["club"] == "Officer" for i in range(num_nodes)], dtype=torch.long
        )
        graph = GraphData(node_features=node_features, edge_index=edge_index, target=labels)
        return [graph]

    def generate_synthetic_small(self) -> List[GraphData]:
        """Generate small synthetic graphs"""
        graphs = []
        for i in range(100):
            num_nodes = np.random.randint(10, 20)
            if i % 3 == 0:
                G = nx.erdos_renyi_graph(num_nodes, 0.3)
                label = 0
            elif i % 3 == 1:
                G = nx.barabasi_albert_graph(num_nodes, 2)
                label = 1
            else:
                G = nx.watts_strogatz_graph(num_nodes, 4, 0.3)
                label = 2
            graph = self._networkx_to_graphdata(G, feature_dim=32, graph_label=label)
            graphs.append(graph)
        return graphs

    def generate_synthetic_medium(self) -> List[GraphData]:
        """Generate medium synthetic graphs with edge features"""
        graphs = []
        for i in range(100):
            num_nodes = np.random.randint(30, 70)
            graph_type = i % 5
            if graph_type == 0:
                G = nx.erdos_renyi_graph(num_nodes, 0.1)
            elif graph_type == 1:
                G = nx.barabasi_albert_graph(num_nodes, 3)
            elif graph_type == 2:
                G = nx.watts_strogatz_graph(num_nodes, 6, 0.3)
            elif graph_type == 3:
                G = nx.powerlaw_cluster_graph(num_nodes, 3, 0.1)
            else:
                G = nx.random_regular_graph(4, num_nodes)
            graph = self._networkx_to_graphdata(
                G, feature_dim=64, edge_feature_dim=8, graph_label=graph_type
            )
            graphs.append(graph)
        return graphs

    def generate_synthetic_large(self) -> List[GraphData]:
        """Generate large synthetic graphs"""
        graphs = []
        for i in range(50):
            num_nodes = np.random.randint(150, 250)
            if i < 25:
                if i % 2 == 0:
                    G = nx.barabasi_albert_graph(num_nodes, 2)
                else:
                    G = nx.random_geometric_graph(num_nodes, 0.1)
            elif i % 2 == 0:
                G = nx.watts_strogatz_graph(num_nodes, 8, 0.3)
            else:
                k = min(10, num_nodes - 1)
                if k % 2 == 1:
                    k -= 1
                G = nx.random_regular_graph(k, num_nodes)
            label = i % 10
            graph = self._networkx_to_graphdata(
                G, feature_dim=128, edge_feature_dim=16, graph_label=label
            )
            graphs.append(graph)
        return graphs

    def generate_grid_graphs(self) -> List[GraphData]:
        """Generate 2D grid graphs"""
        graphs = []
        for i in range(100):
            rows = np.random.randint(3, 8)
            cols = np.random.randint(3, 8)
            G = nx.grid_2d_graph(rows, cols)
            if i % 4 > 0:
                num_extra_edges = np.random.randint(0, 5)
                nodes = list(G.nodes())
                for _ in range(num_extra_edges):
                    u, v = random.sample(nodes, 2)
                    G.add_edge(u, v)
            if rows == cols:
                label = 0
            elif rows > cols:
                label = 1
            else:
                label = 2
            if i % 4 == 3:
                label = 3
            mapping = {node: idx for idx, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            graph = self._networkx_to_graphdata(G, feature_dim=16, graph_label=label)
            graphs.append(graph)
        return graphs

    def generate_tree_graphs(self) -> List[GraphData]:
        """Generate tree-structured graphs"""
        graphs = []
        for i in range(100):
            tree_type = i % 3
            if tree_type == 0:
                height = np.random.randint(3, 5)
                branching = np.random.randint(2, 4)
                G = nx.balanced_tree(branching, height)
            elif tree_type == 1:
                num_nodes = np.random.randint(20, 40)
                G = nx.random_tree(num_nodes)
            else:
                num_nodes = np.random.randint(20, 40)
                G = nx.path_graph(num_nodes)
            graph = self._networkx_to_graphdata(G, feature_dim=32, graph_label=tree_type)
            graphs.append(graph)
        return graphs

    def generate_community_graphs(self) -> List[GraphData]:
        """Generate graphs with community structure"""
        graphs = []
        for i in range(50):
            num_communities = np.random.randint(2, 5)
            community_sizes = [np.random.randint(15, 35) for _ in range(num_communities)]
            p_within = 0.3
            p_between = 0.02
            probs = []
            for i in range(num_communities):
                row = []
                for j in range(num_communities):
                    if i == j:
                        row.append(p_within)
                    else:
                        row.append(p_between)
                probs.append(row)
            G = nx.stochastic_block_model(community_sizes, probs)
            node_labels = []
            for comm_idx, comm_size in enumerate(community_sizes):
                node_labels.extend([comm_idx] * comm_size)
            node_labels = torch.tensor(node_labels, dtype=torch.long)
            num_nodes = G.number_of_nodes()
            node_features = torch.randn(num_nodes, 64)
            edge_list = list(G.edges())
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            graph = GraphData(
                node_features=node_features, edge_index=edge_index, target=node_labels
            )
            graphs.append(graph)
        return graphs

    def _networkx_to_graphdata(
        self,
        G: nx.Graph,
        feature_dim: int,
        edge_feature_dim: Optional[int] = None,
        graph_label: Optional[int] = None,
    ) -> GraphData:
        """Convert NetworkX graph to GraphData"""
        num_nodes = G.number_of_nodes()
        node_features = torch.randn(num_nodes, feature_dim)
        edge_list = list(G.edges())
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
            if edge_feature_dim:
                num_edges = edge_index.size(1)
                edge_attr = torch.randn(num_edges, edge_feature_dim)
            else:
                edge_attr = None
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = None
        if graph_label is not None:
            target = torch.tensor([graph_label], dtype=torch.long)
        else:
            target = None
        return GraphData(
            node_features=node_features, edge_index=edge_index, edge_attr=edge_attr, target=target
        )

    def get_dataset(self, name: str) -> List[GraphData]:
        """
        Get a benchmark dataset by name.
        Args:
            name: Dataset name
        Returns:
            List of graphs
        """
        datasets = {
            "karate_club": self.load_karate_club,
            "synthetic_small": self.generate_synthetic_small,
            "synthetic_medium": self.generate_synthetic_medium,
            "synthetic_large": self.generate_synthetic_large,
            "grid_graphs": self.generate_grid_graphs,
            "tree_graphs": self.generate_tree_graphs,
            "community_graphs": self.generate_community_graphs,
        }
        if name not in datasets:
            raise ValueError(f"Unknown dataset: {name}")
        return datasets[name]()

    def get_all_datasets(self) -> Dict[str, List[GraphData]]:
        """Get all available datasets"""

        all_datasets = {}
        for name in self.get_dataset_info().keys():
            all_datasets[name] = self.get_dataset(name)
        return all_datasets

    def save_dataset(self, name: str, save_dir: str) -> None:
        """
        Save a dataset to disk.
        Args:
            name: Dataset name
            save_dir: Directory to save to
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        graphs = self.get_dataset(name)
        info = self.get_dataset_info()[name]
        with open(save_path / f"{name}_info.json", "w") as f:
            json.dump(
                {
                    "name": info.name,
                    "num_graphs": info.num_graphs,
                    "num_classes": info.num_classes,
                    "avg_nodes": info.avg_nodes,
                    "avg_edges": info.avg_edges,
                    "node_features": info.node_features,
                    "edge_features": info.edge_features,
                    "task_type": info.task_type,
                    "description": info.description,
                },
                f,
                indent=2,
            )
        for i, graph in enumerate(graphs):
            torch.save(
                {
                    "node_features": graph.node_features,
                    "edge_index": graph.edge_index,
                    "edge_attr": graph.edge_attr,
                    "edge_weight": graph.edge_weight,
                    "target": graph.target,
                },
                save_path / f"{name}_graph_{i}.pt",
            )

    def load_dataset_from_disk(self, name: str, load_dir: str) -> List[GraphData]:
        """
        Load a dataset from disk.
        Args:
            name: Dataset name
            load_dir: Directory to load from
        Returns:
            List of graphs
        """
        load_path = Path(load_dir)
        graphs = []
        i = 0
        while True:
            graph_file = load_path / f"{name}_graph_{i}.pt"
            if not graph_file.exists():
                break
            data = torch.load(graph_file)
            graph = GraphData(
                node_features=data["node_features"],
                edge_index=data["edge_index"],
                edge_attr=data.get("edge_attr"),
                edge_weight=data.get("edge_weight"),
                target=data.get("target"),
            )
            graphs.append(graph)
            i += 1
        return graphs


def create_benchmark_splits(
    graphs: List[GraphData],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[GraphData], List[GraphData], List[GraphData]]:
    """
    Split graphs into train/val/test sets.
    Args:
        graphs: List of graphs
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        shuffle: Whether to shuffle before splitting
        seed: Random seed
    Returns:
        Train, validation, and test sets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-06
    if shuffle:
        random.seed(seed)
        graphs = graphs.copy()
        random.shuffle(graphs)
    n = len(graphs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_graphs = graphs[:train_end]
    val_graphs = graphs[train_end:val_end]
    test_graphs = graphs[val_end:]
    return (train_graphs, val_graphs, test_graphs)


if __name__ == "__main__":
    benchmarks = BenchmarkDatasets()
    print("Available datasets:")
    for name, info in benchmarks.get_dataset_info().items():
        print(f"\n{name}:")
        print(f"  Description: {info.description}")
        print(f"  Graphs: {info.num_graphs}")
        print(f"  Avg nodes: {info.avg_nodes}")
        print(f"  Avg edges: {info.avg_edges}")
        print(f"  Task: {info.task_type}")
    print("\nLoading synthetic small dataset...")
    small_graphs = benchmarks.get_dataset("synthetic_small")
    print(f"Loaded {len(small_graphs)} graphs")
    train, val, test = create_benchmark_splits(small_graphs)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
