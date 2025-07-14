import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Parameter, ELU, GELU, Sequential
from torch_geometric.nn import GINConv, EdgeConv, PNAConv, global_max_pool, global_mean_pool, global_add_pool

class GNN(torch.nn.Module):
    def __init__(
        self,
        dim_features: int,
        dim_target: int,
        layers: list = [16, 16, 8],
        degree=None,
        pooling: str = 'max',
        dropout: float = 0.0,
        conv: str = 'GINConv',
        device: str = 'cuda:0',
        gembed: bool = False,
        layer_weighting = False,
        **kwargs
    ):
        """
        Graph Neural Network with configurable layers, pooling, and dynamic weighting.

        Parameters
        ----------
        dim_features : int
            Number of features for each node.
        dim_target : int
            Number of target outputs.
        layers : list, optional
            Hidden layer sizes. Default is [16, 16, 8].
        degree : Tensor, optional
            Degree tensor for PNAConv.
        pooling : str, optional
            Pooling method ('max', 'mean', 'add'). Default is 'max'.
        dropout : float, optional
            Dropout rate. Default is 0.0.
        conv : str, optional
            Type of convolution ('GINConv', 'EdgeConv', 'PNAConv'). Default is 'GINConv'.
        gembed : bool, optional
            If True, the model learns graph embeddings for final classification.
        layer_weighting : bool, optional
            If True, applies learnable layer-wise weighting. Default is True.
        device : str, optional
            Computation device. Default is 'cuda:0'.
        **kwargs : Additional arguments for convolution layers.
        """
        super().__init__()

        self.device = device
        self.dropout = dropout
        self.no_layers = len(layers)
        self.gembed = gembed

        # Select pooling method
        pooling_methods = {'max': global_max_pool, 'mean': global_mean_pool, 'add': global_add_pool}
        self.pooling = pooling_methods.get(pooling, global_max_pool)

        # Initialize layers
        if self.layer_weights:
            self.layer_weights = Parameter(torch.ones(self.no_layers))
        else:
            self.layer_weights = torch.ones(self.no_layers) # no Learning
            
        self.linears = self._initialize_linears(dim_features, dim_target, layers)
        self.convs = self._initialize_convolutions(dim_features, layers, conv, degree, **kwargs)

    def _initialize_linears(self, dim_features, dim_target, layers):
        """Initialize linear layers with batch normalization and ELU activation."""
        return torch.nn.ModuleList([
            Sequential(Linear(dim_features if i == 0 else layers[i - 1], dim_target),
                       BatchNorm1d(dim_target),
                       ELU())
            for i in range(len(layers))
        ])

    def _initialize_convolutions(self, dim_features, layers, conv_type, degree, **kwargs):
        """Initialize convolution layers based on the selected conv type."""
        conv_mapping = {
            'GINConv': lambda in_dim, out_dim: GINConv(
                Sequential(Linear(in_dim, out_dim), BatchNorm1d(out_dim), ELU()), **kwargs
            ),
            'EdgeConv': lambda in_dim, out_dim: EdgeConv(
                Sequential(Linear(2 * in_dim, out_dim), BatchNorm1d(out_dim), ELU()), **kwargs
            ),
            'PNAConv': lambda in_dim, out_dim: PNAConv(
                in_channels=in_dim, out_channels=out_dim,
                aggregators=['mean', 'min', 'max', 'std'],
                scalers=['identity', 'amplification', 'attenuation'],
                deg=degree, edge_dim=0, towers=1, pre_layers=1, post_layers=1, divide_input=False, **kwargs
            )
        }

        if conv_type not in conv_mapping:
            raise NotImplementedError(f"Convolution type '{conv_type}' is not supported.")

        return torch.nn.ModuleList([
            conv_mapping[conv_type](layers[i - 1], layers[i]) for i in range(1, len(layers))
        ])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        Z, out = 0, 0

        # First layer transformation
        z = self.linears[0](x)
        Z += self.layer_weights[0] * z
        out += self.pooling(z, batch)

        for layer in range(1, self.no_layers):
            x = self.convs[layer - 1](x, edge_index)
            z = self.layer_weights[layer] * self.linears[layer](x)
            Z += z
            out += self.pooling(z, batch) if not self.gembed else self.linears[layer](self.pooling(x, batch))

        return F.dropout(out, p=self.dropout, training=self.training), Z, x
