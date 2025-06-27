"""
Graph Neural Network Layer Implementations

This module implements various GNN layers including GCN, GAT, and
    supporting utilities.
Implements the GAT layer from Velickovic et al. (2018).
"""

from enum import Enum
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing  # type: ignore[import-untyped]
from torch_geometric.utils import degree  # type: ignore[import-untyped]
from torch_geometric.utils import add_self_loops


class AggregationType(Enum):
    """Aggregation types for GNN layers"""

    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


class LayerConfig:
    """Configuration for GNN layers"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        aggregation: AggregationType = AggregationType.MEAN,
        dropout: float = 0.0,
        bias: bool = True,
        normalize: bool = True,
        activation: Optional[str] = "relu",
        residual: bool = False,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.aggregation = aggregation
        self.dropout = dropout
        self.bias = bias
        self.normalize = normalize
        self.activation = activation
        self.residual = residual


class GCNLayer(MessagePassing):
    """Graph Convolutional Network layer implementation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        bias: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer parameters"""
        self.lin.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GCN layer"""
        # Add self-loops to the adjacency matrix
        if self.normalize:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linear transformation
        x = self.lin(x)

        # Normalize node features by degree
        if self.normalize:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            return self.propagate(edge_index, x=x, norm=norm)  # type: ignore[no-any-return]
        else:
            return self.propagate(edge_index, x=x)  # type: ignore[no-any-return]

    def message(self, x_j: torch.Tensor, norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Construct messages from neighbors"""
        return norm.view(-1, 1) * x_j if norm is not None else x_j


class GATLayer(MessagePassing):
    """Graph Attention Network layer implementation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer parameters"""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GAT layer"""
        H, C = self.heads, self.out_channels

        # Linear transformation and reshape
        x = self.lin(x).view(-1, H, C)

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, x=x)  # type: ignore[no-any-return]

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_index_i: torch.Tensor
    ) -> torch.Tensor:
        """Construct attention-weighted messages"""
        # Concatenate node features
        alpha = torch.cat([x_i, x_j], dim=-1)  # [E, H, 2*C]

        # Compute attention coefficients
        alpha = (alpha * self.att).sum(dim=-1)  # [E, H]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=1)

        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Update node embeddings"""
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out += self.bias

        return aggr_out


class GNNStack(nn.Module):
    """Stack of GNN layers"""

    def __init__(
        self,
        configs: List[LayerConfig],
        layer_type: str = "gcn",
        final_activation: bool = False,
    ) -> None:
        super().__init__()

        self.configs = configs
        self.layer_type = layer_type.lower()
        self.final_activation = final_activation

        self.layers = nn.ModuleList()

        for i, config in enumerate(configs):
            if self.layer_type == "gcn":
                layer = GCNLayer(
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    bias=config.bias,
                )
            elif self.layer_type == "gat":
                layer = GATLayer(
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    dropout=config.dropout,
                    bias=config.bias,
                )
            else:
                raise ValueError(f"Unknown layer type: {self.layer_type}")

            self.layers.append(layer)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GNN stack"""
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)

            # Apply activation function (except for the last layer unless specified)
            if i < len(self.layers) - 1 or self.final_activation:
                x = F.relu(x)

            # Apply dropout
            if i < len(self.layers) - 1:
                config = self.configs[i]
                if config.dropout > 0:
                    x = F.dropout(x, p=config.dropout, training=self.training)

        return x


def global_add_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Global add pooling"""
    size = int(batch.max().item() + 1)
    return scatter_add(x, batch, dim=0, dim_size=size)


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Global mean pooling"""
    size = int(batch.max().item() + 1)
    return scatter_mean(x, batch, dim=0, dim_size=size)


def global_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Global max pooling"""
    size = int(batch.max().item() + 1)
    return scatter_max(x, batch, dim=0, dim_size=size)[0]


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Scatter add operation"""
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    # Reshape index for scatter_add_
    index = index.view(-1, 1).expand_as(src) if src.dim() > 1 and index.dim() == 1 else index
    return out.scatter_add_(dim, index, src)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Scatter mean operation"""
    out = scatter_add(src, index, dim, dim_size)
    count = scatter_add(torch.ones_like(src), index, dim, dim_size)
    return out / count.clamp(min=1)


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter max operation"""
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    arg_out = torch.zeros(size, dtype=torch.long, device=src.device)
    return out.scatter_reduce_(dim, index.expand_as(src), src, "amax"), arg_out


class SAGELayer(MessagePassing):
    """GraphSAGE layer implementation"""

    def __init__(
        self, in_channels: int, out_channels: int, aggregation: str = "mean", bias: bool = True
    ) -> None:
        """
        Initialize SAGE layer.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            aggregation: Aggregation method ('mean', 'max', 'sum')
            bias: Whether to use bias
        """
        super().__init__(aggr=aggregation)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear transformations
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_neighbor = nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters"""
        self.lin_self.reset_parameters()
        self.lin_neighbor.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Propagate messages
        neighbor_out = self.propagate(edge_index, x=x)

        # Self transformation
        self_out = self.lin_self(x)

        # Combine self and neighbor information
        out = self_out + neighbor_out

        return out  # type: ignore[no-any-return]

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Create messages from neighbors"""

        return self.lin_neighbor(x_j)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


class GINLayer(MessagePassing):
    """
    Graph Isomorphism Network layer.
    Implements the GIN layer from Xu et al. (2019):
    "How Powerful are Graph Neural Networks?"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neural_net: Optional[nn.Module] = None,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps_init = eps

        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer("eps", torch.tensor([eps]))

        if neural_net is None:
            self.neural_net: nn.Module = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
        else:
            self.neural_net = neural_net

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters"""
        self.eps.data.fill_(self.eps_init)
        if hasattr(self.neural_net, "reset_parameters"):
            self.neural_net.reset_parameters()  # type: ignore[operator]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Message passing
        out = self.propagate(edge_index, x=x)

        # Apply neural network
        out = self.neural_net((1 + self.eps) * x + out)

        return out  # type: ignore[no-any-return]

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Create messages from neighbors"""

        return x_j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


class EdgeConvLayer(MessagePassing):
    """
    Edge Convolution layer.
    Implements dynamic graph CNN using edge features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neural_net: Optional[nn.Module] = None,
        aggr: str = "max",
        **kwargs: Any,
    ) -> None:
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if neural_net is None:
            self.neural_net: nn.Module = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
        else:
            self.neural_net = neural_net

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters"""
        if hasattr(self.neural_net, "reset_parameters"):
            self.neural_net.reset_parameters()  # type: ignore[operator]
        else:
            for module in self.neural_net.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()  # type: ignore[operator]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""

        return self.propagate(edge_index, x=x)  # type: ignore[no-any-return]

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Create messages using edge features"""
        # Concatenate source and target node features
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.neural_net(edge_features)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


class ResGNNLayer(nn.Module):
    """
    Residual GNN layer wrapper.
    Adds residual connections to any GNN layer for improved gradient flow.
    """

    def __init__(
        self,
        layer: nn.Module,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.residual: nn.Module = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass with residual connection"""
        identity = self.residual(x)
        out = self.layer(x, *args, **kwargs)
        out = self.dropout(out)
        return out + identity  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


# Example usage
if __name__ == "__main__":
    # Example configuration
    configs = [
        LayerConfig(in_channels=32, out_channels=64),
        LayerConfig(in_channels=64, out_channels=128),
        LayerConfig(in_channels=128, out_channels=64),
    ]
    # Create GNN stack
    model = GNNStack(configs, layer_type="gcn")
    # Example data
    x = torch.randn(100, 32)  # 100 nodes, 32 features
    edge_index = torch.randint(0, 100, (2, 300))  # 300 edges
    # Forward pass
    out = model(x, edge_index)
    print(f"Output shape: {out.shape}")
