# graphcast/model_utils_lite.py
from typing import Optional, Tuple
import numpy as np
from scipy.spatial import transform


def lat_lon_deg_to_spherical(node_lat, node_lon):
    phi = np.deg2rad(node_lon)
    theta = np.deg2rad(90.0 - node_lat)
    return phi, theta


def spherical_to_lat_lon(phi, theta):
    lon = np.mod(np.rad2deg(phi), 360.0)
    lat = 90.0 - np.rad2deg(theta)
    return lat, lon


def cartesian_to_spherical(x, y, z):
    phi = np.arctan2(y, x)
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    return phi, theta


def spherical_to_cartesian(phi, theta):
    return (
        np.cos(phi) * np.sin(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(theta),
    )


def lat_lon_to_cartesian(lat, lon):
    return spherical_to_cartesian(*lat_lon_deg_to_spherical(lat, lon))


def cartesian_to_lat_lon(x, y, z):
    return spherical_to_lat_lon(*cartesian_to_spherical(x, y, z))


def get_rotation_matrices_to_local_coordinates(
    reference_phi,
    reference_theta,
    rotate_latitude: bool,
    rotate_longitude: bool,
):
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

    raise ValueError("At least one of latitude or longitude must be rotated.")


def rotate_with_matrices(rotation_matrices, positions):
    return np.einsum("...ji,...i->...j", rotation_matrices, positions)


def get_relative_position_in_receiver_local_coordinates(
    *,
    node_phi,
    node_theta,
    senders,
    receivers,
    latitude_local_coordinates: bool,
    longitude_local_coordinates: bool,
):
    node_pos = np.stack(
        spherical_to_cartesian(node_phi, node_theta),
        axis=-1,
    )

    if not (latitude_local_coordinates or longitude_local_coordinates):
        return node_pos[senders] - node_pos[receivers]

    rotation_matrices = get_rotation_matrices_to_local_coordinates(
        reference_phi=node_phi,
        reference_theta=node_theta,
        rotate_latitude=latitude_local_coordinates,
        rotate_longitude=longitude_local_coordinates,
    )

    edge_rotation_matrices = rotation_matrices[receivers]

    receiver_pos_rot = rotate_with_matrices(
        edge_rotation_matrices,
        node_pos[receivers],
    )

    sender_pos_rot = rotate_with_matrices(
        edge_rotation_matrices,
        node_pos[senders],
    )

    return sender_pos_rot - receiver_pos_rot


def get_bipartite_relative_position_in_receiver_local_coordinates(
    *,
    senders_node_phi,
    senders_node_theta,
    senders,
    receivers_node_phi,
    receivers_node_theta,
    receivers,
    latitude_local_coordinates: bool,
    longitude_local_coordinates: bool,
):
    senders_node_pos = np.stack(
        spherical_to_cartesian(senders_node_phi, senders_node_theta),
        axis=-1,
    )

    receivers_node_pos = np.stack(
        spherical_to_cartesian(receivers_node_phi, receivers_node_theta),
        axis=-1,
    )

    if not (latitude_local_coordinates or longitude_local_coordinates):
        return senders_node_pos[senders] - receivers_node_pos[receivers]

    receiver_rotation_matrices = get_rotation_matrices_to_local_coordinates(
        reference_phi=receivers_node_phi,
        reference_theta=receivers_node_theta,
        rotate_latitude=latitude_local_coordinates,
        rotate_longitude=longitude_local_coordinates,
    )

    edge_rotation_matrices = receiver_rotation_matrices[receivers]

    receiver_pos_rot = rotate_with_matrices(
        edge_rotation_matrices,
        receivers_node_pos[receivers],
    )

    sender_pos_rot = rotate_with_matrices(
        edge_rotation_matrices,
        senders_node_pos[senders],
    )

    return sender_pos_rot - receiver_pos_rot


def get_graph_spatial_features(
    *,
    node_lat: np.ndarray,
    node_lon: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    add_node_positions: bool,
    add_node_latitude: bool,
    add_node_longitude: bool,
    add_relative_positions: bool,
    edge_normalization_factor: Optional[float] = None,
    relative_longitude_local_coordinates: bool = True,
    relative_latitude_local_coordinates: bool = True,
    sine_cosine_encoding: bool = False,
    encoding_num_freqs: int = 10,
    encoding_multiplicative_factor: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray]:

    num_nodes = node_lat.shape[0]
    num_edges = senders.shape[0]
    dtype = node_lat.dtype

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
        node_features = np.zeros((num_nodes, 0), dtype=dtype)

    edge_features = []

    if add_relative_positions:
        relative_position = get_relative_position_in_receiver_local_coordinates(
            node_phi=node_phi,
            node_theta=node_theta,
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
        edge_features = np.zeros((num_edges, 0), dtype=dtype)

    if sine_cosine_encoding:
        node_features = sine_cosine_transform(
            node_features,
            encoding_num_freqs,
            encoding_multiplicative_factor,
        )
        edge_features = sine_cosine_transform(
            edge_features,
            encoding_num_freqs,
            encoding_multiplicative_factor,
        )

    return node_features, edge_features


def get_bipartite_graph_spatial_features(
    *,
    senders_node_lat: np.ndarray,
    senders_node_lon: np.ndarray,
    senders: np.ndarray,
    receivers_node_lat: np.ndarray,
    receivers_node_lon: np.ndarray,
    receivers: np.ndarray,
    add_node_positions: bool,
    add_node_latitude: bool,
    add_node_longitude: bool,
    add_relative_positions: bool,
    edge_normalization_factor: Optional[float] = None,
    relative_longitude_local_coordinates: bool = True,
    relative_latitude_local_coordinates: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    num_senders = senders_node_lat.shape[0]
    num_receivers = receivers_node_lat.shape[0]
    num_edges = senders.shape[0]
    dtype = senders_node_lat.dtype

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
        senders_node_features.extend(
            spherical_to_cartesian(senders_phi, senders_theta)
        )
        receivers_node_features.extend(
            spherical_to_cartesian(receivers_phi, receivers_theta)
        )

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
        senders_node_features = np.zeros((num_senders, 0), dtype=dtype)
        receivers_node_features = np.zeros((num_receivers, 0), dtype=dtype)

    edge_features = []

    if add_relative_positions:
        relative_position = get_bipartite_relative_position_in_receiver_local_coordinates(
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
        edge_features = np.zeros((num_edges, 0), dtype=dtype)

    return senders_node_features, receivers_node_features, edge_features


def sine_cosine_transform(
    x: np.ndarray,
    num_freqs: int = 10,
    multiplicative_factor: float = 1.2,
) -> np.ndarray:
    freqs = multiplicative_factor ** np.arange(num_freqs)
    phases = freqs * x[..., None]

    x_sin = np.sin(phases)
    x_cos = np.cos(phases)

    x_cat = np.concatenate([x_sin, x_cos], axis=-1)

    return x_cat.reshape((x.shape[0], -1)).astype(np.float32)


def fourier_features(
    values: np.ndarray,
    base_period: float,
    num_frequencies: int,
) -> np.ndarray:
    frequencies = np.arange(1, num_frequencies + 1) / base_period
    angular_frequencies = 2 * np.pi * frequencies

    values_times_angular_freqs = values[..., None] * angular_frequencies

    return np.concatenate(
        [
            np.cos(values_times_angular_freqs),
            np.sin(values_times_angular_freqs),
        ],
        axis=-1,
    ).astype(np.float32)


def flatten_grid_tensor_to_nodes(x):
    """
    x: [B,C,H,W]
    return: [H*W,B,C]
    """
    B, C, H, W = x.shape
    x = x.permute(2, 3, 0, 1).contiguous()
    return x.reshape(H * W, B, C)


def nodes_to_grid_tensor(x, H: int, W: int):
    """
    x: [H*W,B,C]
    return: [B,C,H,W]
    """
    x = x.reshape(H, W, x.shape[1], x.shape[2])
    return x.permute(2, 3, 0, 1).contiguous()