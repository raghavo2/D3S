import open3d as o3d
import numpy as np
 
input_ply = "dust3r_multiview_colored.ply"
output_mesh = "final_presentation_model.obj"
 
print(f"1. Loading DUSt3R point cloud from {input_ply}...")
pcd = o3d.io.read_point_cloud(input_ply)
print(f"Loaded {len(pcd.points)} raw points.")
 
# ---------------------------------------------------------------------------
# STEP 2: Equalize point density (this was skipped before - it's the #1
# cause of "eaten" chunks). DUSt3R clouds are extremely non-uniform: regions
# seen by many overlapping views end up 10-50x denser than regions seen by
# only one view. Poisson uses local density as an implicit confidence
# signal, and the quantile trim in step 6 removes low-density vertices
# first - so sparse-but-real geometry (a corner only one camera saw, a thin
# edge, etc.) is exactly what gets deleted. A light voxel downsample doesn't
# throw away detail, it just merges points already sitting on top of each
# other, so every real surface gets roughly equal "votes" in Poisson.
# ---------------------------------------------------------------------------
print("2. Equalizing point density...")
bbox_diag = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
voxel_size = bbox_diag / 900  # finer grid than before: still equalizes density, keeps more real detail
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"   {len(pcd.points)} points after equalization (voxel={voxel_size:.5f})")
 
# ---------------------------------------------------------------------------
# STEP 3: Gentler, density-aware outlier removal. std_ratio=1.5 is tight
# enough that in sparser (but valid) regions, points get flagged as
# "outliers" purely because their neighbors are farther apart on average -
# not because they're noise. That deletes real geometry before Poisson ever
# sees it. A radius-based pass first (catches true floaters regardless of
# local density) plus a looser statistical pass is much safer.
# ---------------------------------------------------------------------------
print("3. Cleaning noise...")
pcd, _ = pcd.remove_radius_outlier(nb_points=6, radius=voxel_size * 3)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
print(f"   {len(pcd.points)} points after cleaning")
 
# ---------------------------------------------------------------------------
# STEP 4: Normals via hybrid (radius + max-neighbor) search instead of pure
# KNN. Pure KNN=50 forces the same neighbor COUNT everywhere, so near
# corners/edges or in sparser zones it reaches across empty space and mixes
# points from two different surfaces into one blended, wrong normal. Wrong
# normals are the single biggest cause of Poisson leaving a hole instead of
# a surface at that exact spot.
# ---------------------------------------------------------------------------
print("4. Estimating surface normals...")
avg_nn_dist = np.mean(pcd.compute_nearest_neighbor_distance())
pcd.estimate_normals(
    # radius*3 (down from *4): tighter neighborhood = less blending across
    # corners/edges = sharper normals right where crispness lives
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=avg_nn_dist * 3, max_nn=30)
)
# k=50 (up from 30): this only strengthens how robustly orientation is
# propagated across the whole cloud - it does NOT blur individual normals -
# so it's a free way to cut flipped-normal holes without losing sharpness
pcd.orient_normals_consistent_tangent_plane(k=50)
# TIP: if you still have the per-view camera poses from DUSt3R, orienting
# normals toward each camera position (orient_normals_towards_camera_location,
# applied per-view before merging) is more reliable than the line above on
# multi-room / non-convex scenes, where tangent-plane consistency can flip
# normals on isolated clusters.
 
# ---------------------------------------------------------------------------
# STEP 5: Poisson at a more moderate depth. depth=11 pushes resolution high
# enough that any remaining density/normal noise gets amplified into visible
# pits. depth=10 is a solid ceiling for room/object-scale DUSt3R scans -
# crispness should come from clean input (steps 2-4), not from maxing depth.
# linear_fit=True reduces staircasing on flat surfaces like walls/floors.
# scale=1.05 (down from 1.1) keeps the reconstruction domain tighter to your
# actual scan instead of ballooning geometry past its edges.
# ---------------------------------------------------------------------------
print("5. Generating Poisson surface mesh...")
# depth=11 is safe now: it was only dangerous before because it amplified
# bad upstream normals/density into visible pits. With those fixed, higher
# depth just adds real resolution.
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=11, scale=1.05, linear_fit=True
)
densities = np.asarray(densities)
 
# ---------------------------------------------------------------------------
# STEP 6: Trim only the true low-confidence fringe, and report what's
# removed instead of silently nuking it, so you can catch a real chunk at
# risk before it's gone.
# ---------------------------------------------------------------------------
print("6. Post-processing mesh...")
# Sharp corners/edges are naturally LESS locally-planar than a flat wall, so
# Poisson scores them as lower-density even when they're perfectly real.
# A density quantile trim doesn't just remove noise - it disproportionately
# erodes exactly the crisp features you're after. So we trim only the most
# extreme low-confidence fringe (0.3%) and lean on defect-based cleanup
# below for the rest, since defects are identifiable independent of density.
low_density_mask = densities < np.quantile(densities, 0.003)
print(f"   Trimming {low_density_mask.sum()} / {len(densities)} lowest-confidence vertices")
mesh.remove_vertices_by_mask(low_density_mask)
 
# These remove genuine topology defects (zero-area triangles, exact
# duplicates, non-manifold edges) that high-depth Poisson tends to produce -
# none of this is density- or size-based, so it can't mistake a real sharp
# feature for junk.
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_non_manifold_edges()
 
print("   Applying minimal Taubin smoothing...")
# Just enough to settle depth-11 grid noise without rounding off edges.
# If it still looks slightly grainy, nudge to 3-4 - but each extra
# iteration measurably softens corners.
mesh = mesh.filter_smooth_taubin(number_of_iterations=2)
mesh.compute_vertex_normals()
 
# Auto-remove only unambiguous noise specks (a tiny fraction of total
# triangle count - real geometry at depth-11 resolution will always be far
# bigger than this). Everything else is reported, not deleted, so a real
# small chunk never vanishes silently.
print("   Removing stray Poisson noise specks...")
clusters, cluster_tri_count, _ = mesh.cluster_connected_triangles()
clusters = np.asarray(clusters)
cluster_tri_count = np.asarray(cluster_tri_count)
speck_threshold = max(20, int(len(mesh.triangles) * 0.0002))
tiny_clusters = np.where(cluster_tri_count < speck_threshold)[0]
triangles_to_remove = np.isin(clusters, tiny_clusters)
mesh.remove_triangles_by_mask(triangles_to_remove)
mesh.remove_unreferenced_vertices()
print(f"   Removed {len(tiny_clusters)} specks (<{speck_threshold} triangles each, "
      f"{triangles_to_remove.sum()} triangles total)")
kept_sizes = sorted(cluster_tri_count[cluster_tri_count >= speck_threshold], reverse=True)
print(f"   {len(kept_sizes)} components kept, largest 10 (triangle count): {kept_sizes[:10]}")
 
print(f"7. Saving final presentation mesh to {output_mesh}...")
o3d.io.write_triangle_mesh(output_mesh, mesh)
print("Success!")


