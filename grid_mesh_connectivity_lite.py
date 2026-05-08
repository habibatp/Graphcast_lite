# grid_mesh_connectivity_lite.py

import os
import numpy as np
import scipy.spatial
import trimesh

from icosahedral_mesh_lite import TriangularMesh, faces_to_edges


def _grid_lat_lon_to_coordinates(
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
) -> np.ndarray:
    phi_grid, theta_grid = np.meshgrid(
        np.deg2rad(grid_longitude),
        np.deg2rad(90.0 - grid_latitude),
    )

    return np.stack(
        [
            np.cos(phi_grid) * np.sin(theta_grid),
            np.sin(phi_grid) * np.sin(theta_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    ).astype(np.float32)


def radius_query_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: TriangularMesh,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:

    grid_positions = _grid_lat_lon_to_coordinates(
        grid_latitude,
        grid_longitude,
    ).reshape([-1, 3])

    mesh_positions = mesh.vertices.astype(np.float32)

    kd_tree = scipy.spatial.cKDTree(mesh_positions)

    query_indices = kd_tree.query_ball_point(
        x=grid_positions,
        r=radius,
    )

    grid_edge_indices = []
    mesh_edge_indices = []

    for grid_index, mesh_neighbors in enumerate(query_indices):
        if len(mesh_neighbors) == 0:
            _, nearest = kd_tree.query(grid_positions[grid_index], k=1)
            mesh_neighbors = [nearest]

        grid_edge_indices.append(
            np.repeat(grid_index, len(mesh_neighbors))
        )
        mesh_edge_indices.append(mesh_neighbors)

    grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(np.int64)
    mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(np.int64)

    return grid_edge_indices, mesh_edge_indices


def in_mesh_triangle_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: TriangularMesh,
) -> tuple[np.ndarray, np.ndarray]:

    grid_positions = _grid_lat_lon_to_coordinates(
        grid_latitude,
        grid_longitude,
    ).reshape([-1, 3])

    mesh_trimesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        process=False,
    )

    _, _, query_face_indices = trimesh.proximity.closest_point(
        mesh_trimesh,
        grid_positions,
    )

    mesh_edge_indices = mesh.faces[query_face_indices]

    grid_indices = np.arange(grid_positions.shape[0], dtype=np.int64)
    grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])

    mesh_edge_indices = mesh_edge_indices.reshape([-1]).astype(np.int64)
    grid_edge_indices = grid_edge_indices.reshape([-1]).astype(np.int64)

    return grid_edge_indices, mesh_edge_indices


def get_max_edge_distance(mesh: TriangularMesh) -> float:
    senders, receivers = faces_to_edges(mesh.faces)

    sender_pos = mesh.vertices[senders]
    receiver_pos = mesh.vertices[receivers]

    distances = np.linalg.norm(sender_pos - receiver_pos, axis=-1)

    return float(distances.max())


def build_grid_mesh_connectivity(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: TriangularMesh,
    radius_query_fraction_edge_length: float = 0.6,
):
    max_edge_distance = get_max_edge_distance(mesh)

    query_radius = max_edge_distance * radius_query_fraction_edge_length

    grid2mesh_grid_indices, grid2mesh_mesh_indices = radius_query_indices(
        grid_latitude=grid_latitude,
        grid_longitude=grid_longitude,
        mesh=mesh,
        radius=query_radius,
    )

    mesh_senders, mesh_receivers = faces_to_edges(mesh.faces)

    mesh2grid_grid_indices, mesh2grid_mesh_indices = in_mesh_triangle_indices(
        grid_latitude=grid_latitude,
        grid_longitude=grid_longitude,
        mesh=mesh,
    )

    return {
        "query_radius": query_radius,

        "grid2mesh_grid_indices": grid2mesh_grid_indices,
        "grid2mesh_mesh_indices": grid2mesh_mesh_indices,

        "mesh_senders": mesh_senders.astype(np.int64),
        "mesh_receivers": mesh_receivers.astype(np.int64),

        "mesh2grid_grid_indices": mesh2grid_grid_indices,
        "mesh2grid_mesh_indices": mesh2grid_mesh_indices,

        "num_grid_nodes": len(grid_latitude) * len(grid_longitude),
        "num_mesh_nodes": mesh.vertices.shape[0],
    }


def save_connectivity(connectivity: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for key, value in connectivity.items():
        if isinstance(value, np.ndarray):
            np.save(os.path.join(output_dir, f"{key}.npy"), value)

    np.save(
        os.path.join(output_dir, "query_radius.npy"),
        np.array([connectivity["query_radius"]], dtype=np.float32),
    )

    print("✅ Connectivity saved to:", output_dir)


if __name__ == "__main__":
    from icosahedral_mesh_lite import build_mesh

    lat = np.linspace(46.0, 20.0, 64).astype(np.float32)
    lon = np.linspace(-15.0, 6.0, 64).astype(np.float32)

    mesh_data = build_mesh(mesh_size=2)
    mesh = mesh_data["mesh"]

    connectivity = build_grid_mesh_connectivity(
        grid_latitude=lat,
        grid_longitude=lon,
        mesh=mesh,
        radius_query_fraction_edge_length=0.6,
    )

    print("Grid nodes:", connectivity["num_grid_nodes"])
    print("Mesh nodes:", connectivity["num_mesh_nodes"])
    print("Grid2Mesh edges:", len(connectivity["grid2mesh_grid_indices"]))
    print("Mesh edges:", len(connectivity["mesh_senders"]))
    print("Mesh2Grid edges:", len(connectivity["mesh2grid_grid_indices"]))

    save_connectivity(connectivity, "connectivity_cache")