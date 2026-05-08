# graphcast/deep_typed_graph_net_lite.py

from typing import Callable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from graphcast import typed_graph_lite
from graphcast import typed_graph_net_lite


GraphToGraphNetwork = Callable[
    [typed_graph_lite.TypedGraph],
    typed_graph_lite.TypedGraph
]


def get_activation(name: str):
    if name == "relu":
        return nn.ReLU()
    if name == "swish" or name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown activation function: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        activation: str = "silu",
        use_layer_norm: bool = True,
    ):
        super().__init__()

        layers = []
        in_dim = input_size
        act = get_activation(activation)

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(act)
            in_dim = hidden_size

        layers.append(nn.Linear(in_dim, output_size))

        if use_layer_norm:
            layers.append(nn.LayerNorm(output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConcatenatedMLP(nn.Module):
    """
    Equivalent de jraph.concatenated_args(mlp).

    Il concatène les arguments sur la dernière dimension :
        edge_features + sender_features + receiver_features
    ou :
        node_features + received_messages
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        activation: str,
        use_layer_norm: bool,
    ):
        super().__init__()

        self.mlp = MLP(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, *args):
        args = [a for a in args if a is not None]

        if len(args) == 1:
            x = args[0]
        else:
            x = torch.cat(args, dim=-1)

        return self.mlp(x)


class DeepTypedGraphNet(nn.Module):
    """
    Version PyTorch Lite de DeepTypedGraphNet.

    Pipeline :
        1. Embed nodes/edges
        2. Message passing répété
        3. Residual connections
        4. Decoder output
    """

    def __init__(
        self,
        *,
        node_latent_size: Mapping[str, int],
        edge_latent_size: Mapping[str, int],
        mlp_hidden_size: int,
        mlp_num_hidden_layers: int,
        num_message_passing_steps: int,
        num_processor_repetitions: int = 1,
        embed_nodes: bool = True,
        embed_edges: bool = True,
        node_output_size: Optional[Mapping[str, int]] = None,
        edge_output_size: Optional[Mapping[str, int]] = None,
        include_sent_messages_in_node_update: bool = False,
        use_layer_norm: bool = True,
        activation: str = "silu",
        f32_aggregation: bool = False,
        aggregate_normalization: Optional[float] = None,
        name: str = "DeepTypedGraphNetLite",
    ):
        super().__init__()

        self.node_latent_size = dict(node_latent_size)
        self.edge_latent_size = dict(edge_latent_size)
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_num_hidden_layers = mlp_num_hidden_layers
        self.num_message_passing_steps = num_message_passing_steps
        self.num_processor_repetitions = num_processor_repetitions
        self.embed_nodes = embed_nodes
        self.embed_edges = embed_edges
        self.node_output_size = node_output_size
        self.edge_output_size = edge_output_size
        self.include_sent_messages_in_node_update = include_sent_messages_in_node_update
        self.use_layer_norm = use_layer_norm
        self.activation = activation
        self.f32_aggregation = f32_aggregation
        self.aggregate_normalization = aggregate_normalization
        self.name = name

        self._built = False

        self.encoder_node_mlps = nn.ModuleDict()
        self.encoder_edge_mlps = nn.ModuleDict()

        self.processor_edge_mlps = nn.ModuleList()
        self.processor_node_mlps = nn.ModuleList()

        self.decoder_node_mlps = nn.ModuleDict()
        self.decoder_edge_mlps = nn.ModuleDict()

    def build(self, graph_template: typed_graph_lite.TypedGraph):
        """
        Construction paresseuse des MLP selon les dimensions réelles du graphe.
        """

        if self._built:
            return

        # ======================
        # Encoder nodes
        # ======================
        if self.embed_nodes:
            for node_name, node_set in graph_template.nodes.items():
                if node_name not in self.node_latent_size:
                    continue

                input_dim = node_set.features.shape[-1]
                output_dim = self.node_latent_size[node_name]

                self.encoder_node_mlps[node_name] = ConcatenatedMLP(
                    input_size=input_dim,
                    output_size=output_dim,
                    hidden_size=self.mlp_hidden_size,
                    num_hidden_layers=self.mlp_num_hidden_layers,
                    activation=self.activation,
                    use_layer_norm=self.use_layer_norm,
                )

        # ======================
        # Encoder edges
        # ======================
        if self.embed_edges:
            for edge_key, edge_set in graph_template.edges.items():
                edge_name = edge_key.name

                if edge_name not in self.edge_latent_size:
                    continue

                input_dim = edge_set.features.shape[-1]
                output_dim = self.edge_latent_size[edge_name]

                self.encoder_edge_mlps[edge_name] = ConcatenatedMLP(
                    input_size=input_dim,
                    output_size=output_dim,
                    hidden_size=self.mlp_hidden_size,
                    num_hidden_layers=self.mlp_num_hidden_layers,
                    activation=self.activation,
                    use_layer_norm=self.use_layer_norm,
                )

        # ======================
        # Processor steps
        # ======================
        for _ in range(self.num_message_passing_steps):
            step_edge_mlps = nn.ModuleDict()
            step_node_mlps = nn.ModuleDict()

            for edge_key, edge_set in graph_template.edges.items():
                edge_name = edge_key.name
                sender_name, receiver_name = edge_key.node_sets

                edge_dim = self.edge_latent_size[edge_name]
                sender_dim = self.node_latent_size[sender_name]
                receiver_dim = self.node_latent_size[receiver_name]

                input_dim = edge_dim + sender_dim + receiver_dim
                output_dim = edge_dim

                step_edge_mlps[edge_name] = ConcatenatedMLP(
                    input_size=input_dim,
                    output_size=output_dim,
                    hidden_size=self.mlp_hidden_size,
                    num_hidden_layers=self.mlp_num_hidden_layers,
                    activation=self.activation,
                    use_layer_norm=self.use_layer_norm,
                )

            for node_name, node_set in graph_template.nodes.items():
                node_dim = self.node_latent_size[node_name]

                incoming_edge_dims = []
                outgoing_edge_dims = []

                for edge_key in graph_template.edges.keys():
                    edge_name = edge_key.name
                    sender_name, receiver_name = edge_key.node_sets

                    if receiver_name == node_name:
                        incoming_edge_dims.append(self.edge_latent_size[edge_name])

                    if sender_name == node_name:
                        outgoing_edge_dims.append(self.edge_latent_size[edge_name])

                input_dim = node_dim + sum(incoming_edge_dims)

                if self.include_sent_messages_in_node_update:
                    input_dim += sum(outgoing_edge_dims)

                output_dim = node_dim

                step_node_mlps[node_name] = ConcatenatedMLP(
                    input_size=input_dim,
                    output_size=output_dim,
                    hidden_size=self.mlp_hidden_size,
                    num_hidden_layers=self.mlp_num_hidden_layers,
                    activation=self.activation,
                    use_layer_norm=self.use_layer_norm,
                )

            self.processor_edge_mlps.append(step_edge_mlps)
            self.processor_node_mlps.append(step_node_mlps)

        # ======================
        # Decoder nodes
        # ======================
        if self.node_output_size is not None:
            for node_name, output_dim in self.node_output_size.items():
                input_dim = self.node_latent_size[node_name]

                self.decoder_node_mlps[node_name] = ConcatenatedMLP(
                    input_size=input_dim,
                    output_size=output_dim,
                    hidden_size=self.mlp_hidden_size,
                    num_hidden_layers=self.mlp_num_hidden_layers,
                    activation=self.activation,
                    use_layer_norm=False,
                )

        # ======================
        # Decoder edges
        # ======================
        if self.edge_output_size is not None:
            for edge_name, output_dim in self.edge_output_size.items():
                input_dim = self.edge_latent_size[edge_name]

                self.decoder_edge_mlps[edge_name] = ConcatenatedMLP(
                    input_size=input_dim,
                    output_size=output_dim,
                    hidden_size=self.mlp_hidden_size,
                    num_hidden_layers=self.mlp_num_hidden_layers,
                    activation=self.activation,
                    use_layer_norm=False,
                )

        self._built = True

    def forward(
        self,
        input_graph: typed_graph_lite.TypedGraph,
    ) -> typed_graph_lite.TypedGraph:

        if not self._built:
            self.build(input_graph)

        latent_graph_0 = self._embed(input_graph)
        latent_graph_m = self._process(latent_graph_0)
        output_graph = self._output(latent_graph_m)

        return output_graph

    def _embed(
        self,
        input_graph: typed_graph_lite.TypedGraph,
    ) -> typed_graph_lite.TypedGraph:

        embed_edge_fn = None
        embed_node_fn = None

        if self.embed_edges:
            embed_edge_fn = {
                name: mlp
                for name, mlp in self.encoder_edge_mlps.items()
            }

        if self.embed_nodes:
            embed_node_fn = {
                name: mlp
                for name, mlp in self.encoder_node_mlps.items()
            }

        embedder = typed_graph_net_lite.GraphMapFeatures(
            embed_edge_fn=embed_edge_fn,
            embed_node_fn=embed_node_fn,
        )

        return embedder(input_graph)

    def _process(
        self,
        latent_graph_0: typed_graph_lite.TypedGraph,
    ) -> typed_graph_lite.TypedGraph:

        latent_graph = latent_graph_0

        for _ in range(self.num_processor_repetitions):
            for step_i in range(self.num_message_passing_steps):
                latent_graph = self._process_step(
                    latent_graph,
                    step_i,
                )

        return latent_graph

    def _process_step(
        self,
        latent_graph_prev: typed_graph_lite.TypedGraph,
        step_i: int,
    ) -> typed_graph_lite.TypedGraph:

        edge_mlps = self.processor_edge_mlps[step_i]
        node_mlps = self.processor_node_mlps[step_i]

        update_edge_fn = {
            name: mlp
            for name, mlp in edge_mlps.items()
        }

        update_node_fn = {}

        for node_name, mlp in node_mlps.items():

            if self.include_sent_messages_in_node_update:
                update_node_fn[node_name] = self._make_node_fn_with_sent(mlp)
            else:
                update_node_fn[node_name] = self._make_node_fn_received_only(mlp)

        aggregate_fn = typed_graph_net_lite.segment_sum

        if self.aggregate_normalization is not None:
            base_aggregate_fn = aggregate_fn

            def normalized_aggregate(data, segment_ids, num_segments):
                out = base_aggregate_fn(data, segment_ids, num_segments)
                return out / self.aggregate_normalization

            aggregate_fn = normalized_aggregate

        processor_network = typed_graph_net_lite.InteractionNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            aggregate_edges_for_nodes_fn=aggregate_fn,
            include_sent_messages_in_node_update=self.include_sent_messages_in_node_update,
        )

        latent_graph_k = processor_network(latent_graph_prev)

        # Residual connections.
        nodes_with_residuals = {}

        for name, prev_set in latent_graph_prev.nodes.items():
            nodes_with_residuals[name] = prev_set._replace(
                features=prev_set.features + latent_graph_k.nodes[name].features
            )

        edges_with_residuals = {}

        for key, prev_set in latent_graph_prev.edges.items():
            edges_with_residuals[key] = prev_set._replace(
                features=prev_set.features + latent_graph_k.edges[key].features
            )

        return latent_graph_k._replace(
            nodes=nodes_with_residuals,
            edges=edges_with_residuals,
        )

    def _make_node_fn_received_only(self, mlp: nn.Module):
        def node_fn(node_features, received_features):
            messages = []

            for _, value in received_features.items():
                messages.append(value)

            if len(messages) == 0:
                x = node_features
            else:
                x = torch.cat([node_features] + messages, dim=-1)

            return mlp(x)

        return node_fn

    def _make_node_fn_with_sent(self, mlp: nn.Module):
        def node_fn(node_features, sent_features, received_features):
            messages = []

            for _, value in sent_features.items():
                messages.append(value)

            for _, value in received_features.items():
                messages.append(value)

            if len(messages) == 0:
                x = node_features
            else:
                x = torch.cat([node_features] + messages, dim=-1)

            return mlp(x)

        return node_fn

    def _output(
        self,
        latent_graph: typed_graph_lite.TypedGraph,
    ) -> typed_graph_lite.TypedGraph:

        embed_node_fn = None
        embed_edge_fn = None

        if len(self.decoder_node_mlps) > 0:
            embed_node_fn = {
                name: mlp
                for name, mlp in self.decoder_node_mlps.items()
            }

        if len(self.decoder_edge_mlps) > 0:
            embed_edge_fn = {
                name: mlp
                for name, mlp in self.decoder_edge_mlps.items()
            }

        output_network = typed_graph_net_lite.GraphMapFeatures(
            embed_edge_fn=embed_edge_fn,
            embed_node_fn=embed_node_fn,
        )

        return output_network(latent_graph)