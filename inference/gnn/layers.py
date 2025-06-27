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
from torch_geometric.nn import MessagePassing, GCNConv, GATConv, SAGEConv, GINConv, EdgeConv  # type: ignore[import-untyped]
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


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer using PyTorch Geometric"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        bias: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.conv = GCNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            improved=improved,
            cached=cached,
            bias=bias,
            normalize=normalize,
        )

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Access bias parameter from underlying layer"""
        return self.conv.bias

    @property
    def lin(self) -> nn.Module:
        """Access linear layer for compatibility"""
        return self.conv.lin

    def reset_parameters(self) -> None:
        """Reset layer parameters"""
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GCN layer"""
        return self.conv(x, edge_index)  # type: ignore[no-any-return]


class GATLayer(nn.Module):
    """Graph Attention Network layer using PyTorch Geometric"""

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
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_p = dropout

        self.conv = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            bias=bias,
        )

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Access bias parameter from underlying layer"""
        return self.conv.bias

    @property
    def lin(self) -> nn.Module:
        """Access linear layer for compatibility"""
        return self.conv.lin

    def reset_parameters(self) -> None:
        """Reset layer parameters"""
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GAT layer"""
        return self.conv(x, edge_index)  # type: ignore[no-any-return]


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
            elif self.layer_type == "sage":
                layer = SAGELayer(
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    bias=config.bias,
                )
            elif self.layer_type == "gin":
                layer = GINLayer(
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                )
            elif self.layer_type == "edgeconv":
                layer = EdgeConvLayer(
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                )
            else:
                raise ValueError(f"Unknown layer type: {self.layer_type}")

            # Wrap with ResGNNLayer if residual connections are requested
            if config.residual:
                layer = ResGNNLayer(
                    layer=layer,
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    dropout=config.dropout,
                )

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


class SAGELayer(nn.Module):
    """GraphSAGE layer using PyTorch Geometric"""

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
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = SAGEConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr=aggregation,
            bias=bias,
        )

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Access bias parameter from underlying layer"""
        return getattr(self.conv, 'bias', None)

    @property
    def lin(self) -> nn.Module:
        """Access linear layer for compatibility"""
        return getattr(self.conv, 'lin', self.conv)

    def reset_parameters(self) -> None:
        """Reset parameters"""
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        return self.conv(x, edge_index)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


class GINLayer(nn.Module):
    """
    Graph Isomorphism Network layer using PyTorch Geometric.
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
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if neural_net is None:
            neural_net = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )

        self.conv = GINConv(nn=neural_net, eps=eps, train_eps=train_eps)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Access bias parameter from underlying layer"""
        return getattr(self.conv, 'bias', None)

    @property
    def lin(self) -> nn.Module:
        """Access linear layer for compatibility"""
        return getattr(self.conv, 'nn', self.conv)

    def reset_parameters(self) -> None:
        """Reset parameters"""
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.conv(x, edge_index)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


class EdgeConvLayer(nn.Module):
    """
    Edge Convolution layer using PyTorch Geometric.
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
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if neural_net is None:
            neural_net = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )

        self.conv = EdgeConv(nn=neural_net, aggr=aggr)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Access bias parameter from underlying layer"""
        return getattr(self.conv, 'bias', None)

    @property
    def lin(self) -> nn.Module:
        """Access linear layer for compatibility"""
        return getattr(self.conv, 'nn', self.conv)

    def reset_parameters(self) -> None:
        """Reset parameters"""
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.conv(x, edge_index)  # type: ignore[no-any-return]

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
