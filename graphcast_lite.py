# graphcast_lite.py
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from graphcast import icosahedral_mesh_lite
from graphcast import grid_mesh_connectivity_lite
from graphcast import typed_graph_lite
from graphcast import deep_typed_graph_net_lite
from graphcast_lite import predictor_base_lite
from graphcast_lite import losses_lite

@dataclass
class ModelConfig:
    resolution: float = 1.0
    mesh_size: int = 2
    latent_size: int = 64
    gnn_msg_steps: int = 4
    hidden_layers: int = 1
    radius_query_fraction_edge_length: float = 0.6
    mesh2grid_edge_normalization_factor: Optional[float] = None


@dataclass
class TaskConfigLite:
    input_channels: int = 189
    forcing_channels: int = 7
    output_channels: int = 189
    input_steps: int = 2


def lat_lon_deg_to_spherical(lat, lon):
    phi = np.deg2rad(lon)
    theta = np.deg2rad(90.0 - lat)
    return phi, theta


def spherical_to_lat_lon(phi, theta):
    lon = np.rad2deg(phi)
    lat = 90.0 - np.rad2deg(theta)
    return lat, lon


def cartesian_to_spherical(x, y, z):
    hypot_xy = np.hypot(x, y)
    phi = np.arctan2(y, x)
    theta = np.arctan2(hypot_xy, z)
    return phi, theta


def spherical_to_cartesian(phi, theta):
    return [
        np.cos(phi) * np.sin(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(theta),
    ]


def get_rotation_matrices_to_local_coordinates(
    reference_phi,
    reference_theta,
    rotate_latitude=True,
    rotate_longitude=True,
):
    from scipy.spatial import transform

    azimuthal_rotation = -reference_phi
    polar_rotation = -reference_theta + np.pi / 2

    if rotate_longitude and rotate_latitude:
        return transform.Rotation.from_euler(
            "zy",
            np.stack([azimuthal_rotation, polar_rotation], axis=1),
        ).as_matrix()

    if rotate_longitude:
        return transform.Rotation.from_euler(
            "z",
            np.expand_dims(azimuthal_rotation, axis=1),
        ).as_matrix()

    if rotate_latitude:
        return transform.Rotation.from_euler(
            "zyz",
            np.stack(
                [azimuthal_rotation, polar_rotation, -azimuthal_rotation],
                axis=1,
            ),
        ).as_matrix()

    raise ValueError("At least latitude or longitude must be rotated.")


def rotate_with_matrices(rotation_matrices, positions):
    return np.einsum("...ji,...i->...j", rotation_matrices, positions)


def get_relative_position_in_receiver_local_coordinates(
    *,
    senders_node_phi,
    senders_node_theta,
    receivers_node_phi,
    receivers_node_theta,
    senders,
    receivers,
    latitude_local_coordinates=True,
    longitude_local_coordinates=True,
):
    sender_phi = senders_node_phi[senders]
    sender_theta = senders_node_theta[senders]

    receiver_phi = receivers_node_phi[receivers]
    receiver_theta = receivers_node_theta[receivers]

    sender_pos = np.stack(spherical_to_cartesian(sender_phi, sender_theta), axis=-1)
    receiver_pos = np.stack(spherical_to_cartesian(receiver_phi, receiver_theta), axis=-1)

    relative_position = sender_pos - receiver_pos

    rotation_matrices = get_rotation_matrices_to_local_coordinates(
        reference_phi=receiver_phi,
        reference_theta=receiver_theta,
        rotate_latitude=latitude_local_coordinates,
        rotate_longitude=longitude_local_coordinates,
    )

    return rotate_with_matrices(rotation_matrices, relative_position)


def get_graph_spatial_features(
    *,
    node_lat,
    node_lon,
    senders,
    receivers,
    add_node_positions=False,
    add_node_latitude=True,
    add_node_longitude=True,
    add_relative_positions=True,
    relative_longitude_local_coordinates=True,
    relative_latitude_local_coordinates=True,
    edge_normalization_factor=None,
):
    node_phi, node_theta = lat_lon_deg_to_spherical(node_lat, node_lon)

    node_features = []

    if add_node_positions:
        node_features.extend(spherical_to_cartesian(node_phi, node_theta))

    if add_node_latitude:
        node_features.append(np.cos(node_theta))

    if add_node_longitude:
        node_features.append(np.cos(node_phi))
        node_features.append(np.sin(node_phi))

    if node_features:
        node_features = np.stack(node_features, axis=-1).astype(np.float32)
    else:
        node_features = np.zeros((node_lat.shape[0], 0), dtype=np.float32)

    edge_features = []

    if add_relative_positions:
        relative_position = get_relative_position_in_receiver_local_coordinates(
            senders_node_phi=node_phi,
            senders_node_theta=node_theta,
            receivers_node_phi=node_phi,
            receivers_node_theta=node_theta,
            senders=senders,
            receivers=receivers,
            latitude_local_coordinates=relative_latitude_local_coordinates,
            longitude_local_coordinates=relative_longitude_local_coordinates,
        )

        relative_edge_distances = np.linalg.norm(
            relative_position,
            axis=-1,
            keepdims=True,
        )

        if edge_normalization_factor is None:
            edge_normalization_factor = relative_edge_distances.max()

        edge_features.append(relative_edge_distances / edge_normalization_factor)
        edge_features.append(relative_position / edge_normalization_factor)

    if edge_features:
        edge_features = np.concatenate(edge_features, axis=-1).astype(np.float32)
    else:
        edge_features = np.zeros((senders.shape[0], 0), dtype=np.float32)

    return node_features, edge_features


def get_bipartite_graph_spatial_features(
    *,
    senders_node_lat,
    senders_node_lon,
    senders,
    receivers_node_lat,
    receivers_node_lon,
    receivers,
    add_node_positions=False,
    add_node_latitude=True,
    add_node_longitude=True,
    add_relative_positions=True,
    relative_longitude_local_coordinates=True,
    relative_latitude_local_coordinates=True,
    edge_normalization_factor=None,
):
    senders_phi, senders_theta = lat_lon_deg_to_spherical(
        senders_node_lat,
        senders_node_lon,
    )
    receivers_phi, receivers_theta = lat_lon_deg_to_spherical(
        receivers_node_lat,
        receivers_node_lon,
    )

    senders_node_features = []
    receivers_node_features = []

    if add_node_positions:
        senders_node_features.extend(spherical_to_cartesian(senders_phi, senders_theta))
        receivers_node_features.extend(spherical_to_cartesian(receivers_phi, receivers_theta))

    if add_node_latitude:
        senders_node_features.append(np.cos(senders_theta))
        receivers_node_features.append(np.cos(receivers_theta))

    if add_node_longitude:
        senders_node_features.append(np.cos(senders_phi))
        senders_node_features.append(np.sin(senders_phi))
        receivers_node_features.append(np.cos(receivers_phi))
        receivers_node_features.append(np.sin(receivers_phi))

    if senders_node_features:
        senders_node_features = np.stack(senders_node_features, axis=-1).astype(np.float32)
        receivers_node_features = np.stack(receivers_node_features, axis=-1).astype(np.float32)
    else:
        senders_node_features = np.zeros((senders_node_lat.shape[0], 0), dtype=np.float32)
        receivers_node_features = np.zeros((receivers_node_lat.shape[0], 0), dtype=np.float32)

    edge_features = []

    if add_relative_positions:
        relative_position = get_relative_position_in_receiver_local_coordinates(
            senders_node_phi=senders_phi,
            senders_node_theta=senders_theta,
            receivers_node_phi=receivers_phi,
            receivers_node_theta=receivers_theta,
            senders=senders,
            receivers=receivers,
            latitude_local_coordinates=relative_latitude_local_coordinates,
            longitude_local_coordinates=relative_longitude_local_coordinates,
        )

        relative_edge_distances = np.linalg.norm(
            relative_position,
            axis=-1,
            keepdims=True,
        )

        if edge_normalization_factor is None:
            edge_normalization_factor = relative_edge_distances.max()

        edge_features.append(relative_edge_distances / edge_normalization_factor)
        edge_features.append(relative_position / edge_normalization_factor)

    if edge_features:
        edge_features = np.concatenate(edge_features, axis=-1).astype(np.float32)
    else:
        edge_features = np.zeros((senders.shape[0], 0), dtype=np.float32)

    return senders_node_features, receivers_node_features, edge_features


def add_batch_second_axis(data, batch_size, device):
    data = torch.as_tensor(data, dtype=torch.float32, device=device)
    return data[:, None, :].repeat(1, batch_size, 1)


def np_to_torch(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class GraphCastLite(
    predictor_base_lite.Predictor,
    nn.Module,
):
    def __init__(
        self,
        model_config: ModelConfig,
        task_config: TaskConfigLite,
        lat: np.ndarray,
        lon: np.ndarray,
    ):
        super().__init__()
        self.latitude = lat.astype(np.float32)
        self.model_config = model_config
        self.task_config = task_config

        self.lat = lat.astype(np.float32)
        self.lon = lon.astype(np.float32)

        self.spatial_features_kwargs = dict(
            add_node_positions=False,
            add_node_latitude=True,
            add_node_longitude=True,
            add_relative_positions=True,
            relative_longitude_local_coordinates=True,
            relative_latitude_local_coordinates=True,
        )

        self.meshes = icosahedral_mesh_lite.get_hierarchy_of_triangular_meshes_for_sphere(
            splits=model_config.mesh_size
        )

        self.finest_mesh = self.meshes[-1]

        self.query_radius = (
            grid_mesh_connectivity_lite.get_max_edge_distance(self.finest_mesh)
            * model_config.radius_query_fraction_edge_length
        )

        self.mesh2grid_edge_normalization_factor = (
            model_config.mesh2grid_edge_normalization_factor
        )

        self.num_outputs = task_config.output_channels

        self.grid2mesh_gnn = deep_typed_graph_net_lite.DeepTypedGraphNet(
            embed_nodes=True,
            embed_edges=True,
            edge_latent_size=dict(grid2mesh=model_config.latent_size),
            node_latent_size=dict(
                mesh_nodes=model_config.latent_size,
                grid_nodes=model_config.latent_size,
            ),
            mlp_hidden_size=model_config.latent_size,
            mlp_num_hidden_layers=model_config.hidden_layers,
            num_message_passing_steps=1,
            use_layer_norm=True,
            include_sent_messages_in_node_update=False,
            activation="swish",
            f32_aggregation=True,
            aggregate_normalization=None,
            name="grid2mesh_gnn",
        )

        self.mesh_gnn = deep_typed_graph_net_lite.DeepTypedGraphNet(
            embed_nodes=False,
            embed_edges=True,
            node_latent_size=dict(mesh_nodes=model_config.latent_size),
            edge_latent_size=dict(mesh=model_config.latent_size),
            mlp_hidden_size=model_config.latent_size,
            mlp_num_hidden_layers=model_config.hidden_layers,
            num_message_passing_steps=model_config.gnn_msg_steps,
            use_layer_norm=True,
            include_sent_messages_in_node_update=False,
            activation="swish",
            f32_aggregation=False,
            name="mesh_gnn",
        )

        self.mesh2grid_gnn = deep_typed_graph_net_lite.DeepTypedGraphNet(
            node_output_size=dict(grid_nodes=self.num_outputs),
            embed_nodes=False,
            embed_edges=True,
            edge_latent_size=dict(mesh2grid=model_config.latent_size),
            node_latent_size=dict(
                mesh_nodes=model_config.latent_size,
                grid_nodes=model_config.latent_size,
            ),
            mlp_hidden_size=model_config.latent_size,
            mlp_num_hidden_layers=model_config.hidden_layers,
            num_message_passing_steps=1,
            use_layer_norm=True,
            include_sent_messages_in_node_update=False,
            activation="swish",
            f32_aggregation=False,
            name="mesh2grid_gnn",
        )

        self._init_static_graphs()

    def _init_static_graphs(self):
        self._init_mesh_properties()
        self._init_grid_properties()
        self.grid2mesh_graph_structure = self._init_grid2mesh_graph()
        self.mesh_graph_structure = self._init_mesh_graph()
        self.mesh2grid_graph_structure = self._init_mesh2grid_graph()

    def _init_mesh_properties(self):
        self.num_mesh_nodes = self.finest_mesh.vertices.shape[0]

        mesh_phi, mesh_theta = cartesian_to_spherical(
            self.finest_mesh.vertices[:, 0],
            self.finest_mesh.vertices[:, 1],
            self.finest_mesh.vertices[:, 2],
        )

        mesh_lat, mesh_lon = spherical_to_lat_lon(mesh_phi, mesh_theta)

        self.mesh_nodes_lat = mesh_lat.astype(np.float32)
        self.mesh_nodes_lon = mesh_lon.astype(np.float32)

    def _init_grid_properties(self):
        self.grid_lat = self.lat
        self.grid_lon = self.lon

        self.num_grid_nodes = self.grid_lat.shape[0] * self.grid_lon.shape[0]

        grid_nodes_lon, grid_nodes_lat = np.meshgrid(self.grid_lon, self.grid_lat)

        self.grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
        self.grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)

        self.H = len(self.grid_lat)
        self.W = len(self.grid_lon)

    def _init_grid2mesh_graph(self):
        grid_indices, mesh_indices = grid_mesh_connectivity_lite.radius_query_indices(
            grid_latitude=self.grid_lat,
            grid_longitude=self.grid_lon,
            mesh=self.finest_mesh,
            radius=self.query_radius,
        )

        senders = grid_indices
        receivers = mesh_indices

        senders_node_features, receivers_node_features, edge_features = (
            get_bipartite_graph_spatial_features(
                senders_node_lat=self.grid_nodes_lat,
                senders_node_lon=self.grid_nodes_lon,
                receivers_node_lat=self.mesh_nodes_lat,
                receivers_node_lon=self.mesh_nodes_lon,
                senders=senders,
                receivers=receivers,
                edge_normalization_factor=None,
                **self.spatial_features_kwargs,
            )
        )

        grid_node_set = typed_graph_lite.NodeSet(
            n_node=torch.tensor([self.num_grid_nodes], dtype=torch.long),
            features=torch.tensor(senders_node_features, dtype=torch.float32),
        )

        mesh_node_set = typed_graph_lite.NodeSet(
            n_node=torch.tensor([self.num_mesh_nodes], dtype=torch.long),
            features=torch.tensor(receivers_node_features, dtype=torch.float32),
        )

        edge_set = typed_graph_lite.EdgeSet(
            n_edge=torch.tensor([senders.shape[0]], dtype=torch.long),
            indices=typed_graph_lite.EdgesIndices(
                senders=torch.tensor(senders, dtype=torch.long),
                receivers=torch.tensor(receivers, dtype=torch.long),
            ),
            features=torch.tensor(edge_features, dtype=torch.float32),
        )

        return typed_graph_lite.TypedGraph(
            context=typed_graph_lite.Context(
                n_graph=torch.tensor([1], dtype=torch.long),
                features=torch.zeros(1, 0),
            ),
            nodes={
                "grid_nodes": grid_node_set,
                "mesh_nodes": mesh_node_set,
            },
            edges={
                typed_graph_lite.EdgeSetKey(
                    "grid2mesh",
                    ("grid_nodes", "mesh_nodes"),
                ): edge_set
            },
        )

    def _init_mesh_graph(self):
        merged_mesh = icosahedral_mesh_lite.merge_meshes(self.meshes)

        senders, receivers = icosahedral_mesh_lite.faces_to_edges(merged_mesh.faces)

        node_features, edge_features = get_graph_spatial_features(
            node_lat=self.mesh_nodes_lat,
            node_lon=self.mesh_nodes_lon,
            senders=senders,
            receivers=receivers,
            **self.spatial_features_kwargs,
        )

        mesh_node_set = typed_graph_lite.NodeSet(
            n_node=torch.tensor([self.num_mesh_nodes], dtype=torch.long),
            features=torch.tensor(node_features, dtype=torch.float32),
        )

        edge_set = typed_graph_lite.EdgeSet(
            n_edge=torch.tensor([senders.shape[0]], dtype=torch.long),
            indices=typed_graph_lite.EdgesIndices(
                senders=torch.tensor(senders, dtype=torch.long),
                receivers=torch.tensor(receivers, dtype=torch.long),
            ),
            features=torch.tensor(edge_features, dtype=torch.float32),
        )

        return typed_graph_lite.TypedGraph(
            context=typed_graph_lite.Context(
                n_graph=torch.tensor([1], dtype=torch.long),
                features=torch.zeros(1, 0),
            ),
            nodes={"mesh_nodes": mesh_node_set},
            edges={
                typed_graph_lite.EdgeSetKey(
                    "mesh",
                    ("mesh_nodes", "mesh_nodes"),
                ): edge_set
            },
        )

    def _init_mesh2grid_graph(self):
        grid_indices, mesh_indices = grid_mesh_connectivity_lite.in_mesh_triangle_indices(
            grid_latitude=self.grid_lat,
            grid_longitude=self.grid_lon,
            mesh=self.finest_mesh,
        )

        senders = mesh_indices
        receivers = grid_indices

        senders_node_features, receivers_node_features, edge_features = (
            get_bipartite_graph_spatial_features(
                senders_node_lat=self.mesh_nodes_lat,
                senders_node_lon=self.mesh_nodes_lon,
                receivers_node_lat=self.grid_nodes_lat,
                receivers_node_lon=self.grid_nodes_lon,
                senders=senders,
                receivers=receivers,
                edge_normalization_factor=self.mesh2grid_edge_normalization_factor,
                **self.spatial_features_kwargs,
            )
        )

        grid_node_set = typed_graph_lite.NodeSet(
            n_node=torch.tensor([self.num_grid_nodes], dtype=torch.long),
            features=torch.tensor(receivers_node_features, dtype=torch.float32),
        )

        mesh_node_set = typed_graph_lite.NodeSet(
            n_node=torch.tensor([self.num_mesh_nodes], dtype=torch.long),
            features=torch.tensor(senders_node_features, dtype=torch.float32),
        )

        edge_set = typed_graph_lite.EdgeSet(
            n_edge=torch.tensor([senders.shape[0]], dtype=torch.long),
            indices=typed_graph_lite.EdgesIndices(
                senders=torch.tensor(senders, dtype=torch.long),
                receivers=torch.tensor(receivers, dtype=torch.long),
            ),
            features=torch.tensor(edge_features, dtype=torch.float32),
        )

        return typed_graph_lite.TypedGraph(
            context=typed_graph_lite.Context(
                n_graph=torch.tensor([1], dtype=torch.long),
                features=torch.zeros(1, 0),
            ),
            nodes={
                "grid_nodes": grid_node_set,
                "mesh_nodes": mesh_node_set,
            },
            edges={
                typed_graph_lite.EdgeSetKey(
                    "mesh2grid",
                    ("mesh_nodes", "grid_nodes"),
                ): edge_set
            },
        )

    def forward(self, inputs, forcings=None):
        """
        inputs:
            [B, T, C, H, W]

        forcings:
            [B, F, H, W]

        output:
            [B, C_out, H, W]
        """
        B, T, C, H, W = inputs.shape

        assert H == self.H and W == self.W

        grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)

        latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn(
            grid_node_features
        )

        updated_latent_mesh_nodes = self._run_mesh_gnn(
            latent_mesh_nodes
        )

        output_grid_nodes = self._run_mesh2grid_gnn(
            updated_latent_mesh_nodes,
            latent_grid_nodes,
        )

        return self._grid_node_outputs_to_prediction(output_grid_nodes)

    def _inputs_to_grid_node_features(self, inputs, forcings=None):
        """
        inputs:
            [B,T,C,H,W]

        forcings:
            [B,F,H,W]

        return:
            [num_grid_nodes,B,channels]
        """
        B, T, C, H, W = inputs.shape

        x = inputs.reshape(B, T * C, H, W)

        if forcings is not None:
            x = torch.cat([x, forcings], dim=1)

        x = x.permute(2, 3, 0, 1).contiguous()
        x = x.reshape(H * W, B, -1)

        return x

    def _grid_node_outputs_to_prediction(self, grid_node_outputs):
        """
        grid_node_outputs:
            [num_grid_nodes,B,C_out]

        return:
            [B,C_out,H,W]
        """
        x = grid_node_outputs.reshape(self.H, self.W, grid_node_outputs.shape[1], -1)
        x = x.permute(2, 3, 0, 1).contiguous()
        return x

    def _move_graph_to_device(self, graph, device):
        nodes = {}
        for name, node_set in graph.nodes.items():
            nodes[name] = node_set._replace(
                n_node=node_set.n_node.to(device),
                features=node_set.features.to(device),
            )

        edges = {}
        for key, edge_set in graph.edges.items():
            edges[key] = edge_set._replace(
                n_edge=edge_set.n_edge.to(device),
                indices=typed_graph_lite.EdgesIndices(
                    senders=edge_set.indices.senders.to(device),
                    receivers=edge_set.indices.receivers.to(device),
                ),
                features=edge_set.features.to(device),
            )

        context = graph.context._replace(
            n_graph=graph.context.n_graph.to(device),
            features=graph.context.features.to(device),
        )

        return graph._replace(nodes=nodes, edges=edges, context=context)

    def _run_grid2mesh_gnn(self, grid_node_features):
        device = grid_node_features.device
        B = grid_node_features.shape[1]

        graph = self._move_graph_to_device(self.grid2mesh_graph_structure, device)

        grid_nodes = graph.nodes["grid_nodes"]
        mesh_nodes = graph.nodes["mesh_nodes"]

        grid_struct = add_batch_second_axis(grid_nodes.features, B, device)
        mesh_struct = add_batch_second_axis(mesh_nodes.features, B, device)

        new_grid_nodes = grid_nodes._replace(
            features=torch.cat([grid_node_features, grid_struct], dim=-1)
        )

        dummy_mesh_node_features = torch.zeros(
            self.num_mesh_nodes,
            B,
            grid_node_features.shape[-1],
            dtype=grid_node_features.dtype,
            device=device,
        )

        new_mesh_nodes = mesh_nodes._replace(
            features=torch.cat([dummy_mesh_node_features, mesh_struct], dim=-1)
        )

        edge_key = graph.edge_key_by_name("grid2mesh")
        edges = graph.edges[edge_key]

        new_edges = edges._replace(
            features=add_batch_second_axis(edges.features, B, device)
        )

        input_graph = graph._replace(
            nodes={
                "grid_nodes": new_grid_nodes,
                "mesh_nodes": new_mesh_nodes,
            },
            edges={edge_key: new_edges},
        )

        out = self.grid2mesh_gnn(input_graph)

        return (
            out.nodes["mesh_nodes"].features,
            out.nodes["grid_nodes"].features,
        )

    def _run_mesh_gnn(self, latent_mesh_nodes):
        device = latent_mesh_nodes.device
        B = latent_mesh_nodes.shape[1]

        graph = self._move_graph_to_device(self.mesh_graph_structure, device)

        edge_key = graph.edge_key_by_name("mesh")
        edges = graph.edges[edge_key]

        new_edges = edges._replace(
            features=add_batch_second_axis(edges.features, B, device)
        )

        nodes = graph.nodes["mesh_nodes"]._replace(
            features=latent_mesh_nodes
        )

        input_graph = graph._replace(
            nodes={"mesh_nodes": nodes},
            edges={edge_key: new_edges},
        )

        out = self.mesh_gnn(input_graph)

        return out.nodes["mesh_nodes"].features

    def _run_mesh2grid_gnn(self, updated_latent_mesh_nodes, latent_grid_nodes):
        device = updated_latent_mesh_nodes.device
        B = updated_latent_mesh_nodes.shape[1]

        graph = self._move_graph_to_device(self.mesh2grid_graph_structure, device)

        mesh_nodes = graph.nodes["mesh_nodes"]._replace(
            features=updated_latent_mesh_nodes
        )

        grid_nodes = graph.nodes["grid_nodes"]._replace(
            features=latent_grid_nodes
        )

        edge_key = graph.edge_key_by_name("mesh2grid")
        edges = graph.edges[edge_key]

        new_edges = edges._replace(
            features=add_batch_second_axis(edges.features, B, device)
        )

        input_graph = graph._replace(
            nodes={
                "mesh_nodes": mesh_nodes,
                "grid_nodes": grid_nodes,
            },
            edges={edge_key: new_edges},
        )

        out = self.mesh2grid_gnn(input_graph)

        return out.nodes["grid_nodes"].features
    def loss(
        self,
        inputs,
        targets,
        forcings=None,
        channel_weights=None,
    ):
        predictions = self.forward(inputs, forcings)

        return losses_lite.weighted_mse(
            predictions=predictions,
            targets=targets,
            latitude=self.latitude,
            channel_weights=channel_weights,
        )
    def loss_and_predictions(
        self,
        inputs,
        targets,
        forcings=None,
        channel_weights=None,
    ):
        predictions = self.forward(inputs, forcings)

        loss, diagnostics = losses_lite.weighted_mse(
            predictions=predictions,
            targets=targets,
            latitude=self.latitude,
            channel_weights=channel_weights,
        )

        return (loss, diagnostics), predictions

if __name__ == "__main__":
    lat = np.linspace(46.0, 20.0, 64).astype(np.float32)
    lon = np.linspace(-15.0, 6.0, 64).astype(np.float32)

    model_config = ModelConfig(
        resolution=1.0,
        mesh_size=2,
        latent_size=64,
        gnn_msg_steps=4,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6,
    )

    task_config = TaskConfigLite(
        input_channels=189,
        forcing_channels=7,
        output_channels=189,
        input_steps=2,
    )

    model = GraphCastLite(
        model_config=model_config,
        task_config=task_config,
        lat=lat,
        lon=lon,
    )

    x = torch.randn(1, 2, 189, 64, 64)
    f = torch.randn(1, 7, 64, 64)

    y = model(x, f)

    print("Output:", y.shape)
