"""
Deliverable export: GLB (web viewer) and LAS (GIS integration).
Adapted from export_deliverables.py — original script is preserved unchanged.
"""

import os


def export_deliverables(input_ply, input_obj, output_dir, on_log=None):
    """
    Convert reconstruction outputs into industry-standard deliverables.

    - `.glb` — Web-optimized 3D mesh for browser viewers
    - `.las` — GIS-ready georeferenced point cloud

    Args:
        input_ply: Path to the colored point cloud (.ply).
        input_obj: Path to the surface mesh (.obj).
        output_dir: Directory for deliverable outputs.
        on_log: Optional progress callback.

    Returns:
        dict with keys: glb_path (str|None), las_path (str|None)
    """
    log = on_log or (lambda msg: None)
    os.makedirs(output_dir, exist_ok=True)

    result = {"glb_path": None, "las_path": None}

    # --- 1. Export GLB ---
    log("Converting mesh to web-optimized GLB…")
    if os.path.isfile(input_obj):
        try:
            import trimesh

            mesh = trimesh.load(input_obj, process=False)
            glb_path = os.path.join(output_dir, "model.glb")
            mesh.export(glb_path)
            result["glb_path"] = glb_path
            log(f"  Saved GLB: {glb_path}")
        except Exception as e:
            log(f"  GLB export error: {e}")
    else:
        log(f"  Skipping GLB — mesh not found: {input_obj}")

    # --- 2. Export LAS ---
    log("Converting point cloud to LAS format…")
    if os.path.isfile(input_ply):
        try:
            import numpy as np
            import open3d as o3d
            import laspy

            pcd = o3d.io.read_point_cloud(input_ply)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) * 65535  # 16-bit color for LAS

            header = laspy.LasHeader(point_format=2, version="1.2")
            las = laspy.LasData(header)
            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]

            if colors.size > 0:
                las.red = colors[:, 0].astype(np.uint16)
                las.green = colors[:, 1].astype(np.uint16)
                las.blue = colors[:, 2].astype(np.uint16)

            las_path = os.path.join(output_dir, "point_cloud.las")
            las.write(las_path)
            result["las_path"] = las_path
            log(f"  Saved LAS: {las_path}")
        except Exception as e:
            log(f"  LAS export error: {e}")
    else:
        log(f"  Skipping LAS — point cloud not found: {input_ply}")

    log("Deliverable export complete.")
    return result
