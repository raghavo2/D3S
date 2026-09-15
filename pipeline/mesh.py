"""
Poisson surface reconstruction from point cloud to mesh.
Adapted from process_dust3r.py — original script is preserved unchanged.
"""

import os


def generate_mesh(input_ply, output_obj, on_log=None):
    """
    Generate a high-fidelity mesh from a colored point cloud using Screened
    Poisson Surface Reconstruction with Taubin smoothing.

    The pipeline follows the proven approach from process_dust3r.py:
    1. Voxel downsampling to equalize density
    2. Density-aware outlier removal
    3. Hybrid normal estimation
    4. Poisson reconstruction (depth 11)
    5. Low-confidence fringe trimming
    6. Topology cleanup + Taubin smoothing
    7. Noise speck removal

    Args:
        input_ply: Path to the input colored point cloud (.ply).
        output_obj: Path for the output mesh (.obj).
        on_log: Optional progress callback.

    Returns:
        dict with keys: vertex_count, triangle_count, output_obj
    """
    import numpy as np
    import open3d as o3d

    log = on_log or (lambda msg: None)

    if not os.path.isfile(input_ply):
        raise FileNotFoundError(f"Point cloud not found: {input_ply}")

    # --- Step 1: Load ---
    log(f"Loading point cloud from {input_ply}…")
    pcd = o3d.io.read_point_cloud(input_ply)
    log(f"Loaded {len(pcd.points)} raw points.")

    # --- Step 2: Equalize density ---
    log("Equalizing point density…")
    bbox_diag = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
    voxel_size = bbox_diag / 900
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    log(f"  {len(pcd.points)} points after equalization (voxel={voxel_size:.5f})")

    # --- Step 3: Outlier removal ---
    log("Cleaning noise…")
    pcd, _ = pcd.remove_radius_outlier(nb_points=6, radius=voxel_size * 3)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    log(f"  {len(pcd.points)} points after cleaning")

    # --- Step 4: Normal estimation ---
    log("Estimating surface normals…")
    avg_nn_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=avg_nn_dist * 3, max_nn=30
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=50)
    log("Normals estimated and oriented.")

    # --- Step 5: Poisson reconstruction ---
    log("Generating Poisson surface mesh (depth=11)…")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=11, scale=1.05, linear_fit=True
    )
    densities = np.asarray(densities)
    log(f"  Raw mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    # --- Step 6: Post-processing ---
    log("Post-processing mesh…")

    # Trim lowest-confidence fringe (0.3%)
    low_density_mask = densities < np.quantile(densities, 0.003)
    log(f"  Trimming {low_density_mask.sum()} lowest-confidence vertices")
    mesh.remove_vertices_by_mask(low_density_mask)

    # Topology cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Taubin smoothing
    log("  Applying Taubin smoothing…")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=2)
    mesh.compute_vertex_normals()

    # --- Step 7: Remove noise specks ---
    log("Removing stray noise specks…")
    clusters, cluster_tri_count, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    cluster_tri_count = np.asarray(cluster_tri_count)
    speck_threshold = max(20, int(len(mesh.triangles) * 0.0002))
    tiny_clusters = np.where(cluster_tri_count < speck_threshold)[0]
    triangles_to_remove = np.isin(clusters, tiny_clusters)
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    log(
        f"  Removed {len(tiny_clusters)} specks "
        f"({triangles_to_remove.sum()} triangles)"
    )

    kept_sizes = sorted(
        cluster_tri_count[cluster_tri_count >= speck_threshold], reverse=True
    )
    log(f"  {len(kept_sizes)} components kept, largest: {kept_sizes[:5]}")

    # --- Save ---
    os.makedirs(os.path.dirname(output_obj) or ".", exist_ok=True)
    log(f"Saving mesh to {output_obj}…")
    o3d.io.write_triangle_mesh(output_obj, mesh)

    result = {
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles),
        "output_obj": output_obj,
    }
    log(f"Done — {result['vertex_count']} vertices, {result['triangle_count']} triangles")
    return result
