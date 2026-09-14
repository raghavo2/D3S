import os
import numpy as np
import trimesh
import laspy
import open3d as o3d

# Input paths (based on your current files)
input_ply = "dust3r_multiview_colored.ply"
input_obj = "final_presentation_model.obj"
output_dir = "drone_deliverables"

os.makedirs(output_dir, exist_ok=True)
print("=" * 60)
print("GENERATING ADDITIONAL PROJECT DELIVERABLES")
print("=" * 60)

# --------------------------------------------------
# 1. Export Web-Optimized GLB / GLTF (.glb / .gltf)
# --------------------------------------------------
print("\n[1/2] Converting mesh to Web-Optimized GLB/GLTF...")
if os.path.exists(input_obj):
    mesh = trimesh.load(input_obj, process=False)
    
    glb_path = os.path.join(output_dir, "model_viewer.glb")
    mesh.export(glb_path)
    print(f" -> Saved Web 3D Asset: {glb_path}")
else:
    print(f"Skipping GLB: {input_obj} not found.")

# --------------------------------------------------
# 2. Export GIS-Ready Point Cloud (.las)
# --------------------------------------------------
print("\n[2/2] Converting Point Cloud to LAS format...")
if os.path.exists(input_ply):
    pcd = o3d.io.read_point_cloud(input_ply)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 65535  # LAS expects 16-bit color channels (0-65535)

    # Create a new LAS file (Point Format 2 includes RGB colors)
    header = laspy.LasHeader(point_format=2, version="1.2")
    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    if colors.size > 0:
        las.red = colors[:, 0].astype(np.uint16)
        las.green = colors[:, 1].astype(np.uint16)
        las.blue = colors[:, 2].astype(np.uint16)

    las_path = os.path.join(output_dir, "point_cloud_gis.las")
    las.write(las_path)
    print(f" -> Saved GIS Point Cloud: {las_path}")
else:
    print(f"Skipping LAS: {input_ply} not found.")


print("\n" + "=" * 60)
print(f"All available deliverables successfully compiled in: {output_dir}/")
print("=" * 60)