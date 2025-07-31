
# Tensile_Bar_Generator.py

# Worked
import trimesh
import shapely.geometry as geom
import shapely.affinity as affinity
import numpy as np
import os
import cv2                     # ← add this
from PIL import Image


def build_astm_by_block_with_solid_ends(
    binary_img:   np.ndarray,  # 50×50 array, 0 or 255
    stl_path:     str,
    block_size:   float = 3.0,   # mm: the width/height of one block
    thickness:    float = 3.2
):
    """
    Tile whole-block shapes across the gauge region, then add solid grips.
    """
    # ASTM Type I dims
    L_total, W_end, L_gauge, W_gauge = 165.0, 19.0, 50.0, 13.0
    grip_len = (L_total - L_gauge)/2.0
    half_th  = thickness/2.0

    # 1) Extract one “block” polygon from your binary image
    #    (same as before in build_astm_d638_lattice_by_block)
   
    mask = (binary_img > 128).astype(np.uint8)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)  # Dilation to fill gaps

 
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    block_polys = []

    for cnt in contours:
        pts = cnt.squeeze() * (block_size / 50.0)  # scale 50px→block_size
        if len(pts) < 3: continue
        block_polys.append(geom.Polygon(pts))
    block_shape = geom.GeometryCollection(block_polys).buffer(0.035)

    # 2) How many whole blocks fit across gauge
    # tile exactly as many whole cells as will fit:
    nx = int(round(L_gauge / block_size))
    ny = int(round(W_gauge / block_size))

    # nx = int(np.floor(L_gauge   / block_size))    
    # ny = int(np.floor(W_gauge   / block_size))
    
    # gauge region sits in the middle of the 19 mm height
    gauge_y0 = (W_end - W_gauge)/2.0
    # center the discrete rows inside that 13 mm band
    y_off    = gauge_y0 + (W_gauge - ny*block_size)/2.0


    meshes = []
    # 3) Tile blocks in gauge only
    for i in range(nx):
        for j in range(ny):
            # footprint in mm
            tx = grip_len + i * block_size # + (block_size / 2)
            ty = j * block_size + y_off # + (block_size / 2)
            block_pos = affinity.translate(block_shape, xoff=tx, yoff=ty)
            mesh = trimesh.creation.extrude_polygon(block_pos, thickness, engine="triangle")
            meshes.append(mesh)

    # 4) Solid grips
    left  = trimesh.creation.box(extents=(grip_len, W_end, thickness))
    right = trimesh.creation.box(extents=(grip_len, W_end, thickness))
    left.apply_translation((grip_len/2,       W_end/2, half_th))
    right.apply_translation((grip_len + L_gauge + grip_len/2, W_end/2, half_th))
    meshes.extend([left, right])

    # 5) Combine & export
    final = trimesh.util.concatenate(meshes)
    final.export(stl_path)
    
    print(f" Exported lattice‐in‐middle bar → {stl_path}")
        

if __name__ == "__main__":

    # 1) load your binary lattice image
    img    = Image.open("vae_output_sample34_binary.png").convert("L")
    binary = np.array(img)  # 50×50, 0 or 255

    # 2) choose block size = gauge_length / 50 px = 1 mm
    block_size = 3.0   

    # 3) build and export
    build_astm_by_block_with_solid_ends(
        binary_img = binary,
        stl_path   = "constructed_bar_sample34.stl",
        block_size = block_size,
        thickness  = 3.2
        )
