# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 14:02:56 2025

@author: joory
"""

import shapely.affinity as affinity


import h5py
import math
import numpy as np
import cv2
from PIL import Image
import shapely.geometry as geom
import trimesh
import os
        
def build_astm_d638_lattice_by_block(
    img_array: np.ndarray,
    stl_path:   str,
    pixel_size: float = 0.1,
    threshold:  int   = 128,
    thickness:  float = 3.2
):
    # 1) make sure img_array is uint8
    arr = img_array
    if arr.dtype != np.uint8:
        arr = ((arr - arr.min())/(arr.max()-arr.min())*255).astype(np.uint8)

    H, W = arr.shape
    block_w_mm = W * pixel_size
    block_h_mm = H * pixel_size

    # 2) extract the single-block polygon in mm
    mask = (arr > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    base_polys = []
    for cnt in contours:
        pts = cnt.squeeze() * pixel_size
        if len(pts) < 3: continue
        base_polys.append(geom.Polygon(pts))
    block_poly = geom.GeometryCollection(base_polys).buffer(0)

    # 3) build ASTM bar footprint in mm
    L, We, Lg, Wg = 165, 19, 50, 13
    end1 = geom.box(0, 0, (L - Lg)/2, We)
    mid  = geom.box((L - Lg)/2, (We - Wg)/2, (L + Lg)/2, (We + Wg)/2)
    end2 = geom.box((L + Lg)/2, 0, L, We)
    bar_poly = end1.union(mid).union(end2)

    # 4) how many whole blocks fit in X & Y
    nx = int(math.floor(L  / block_w_mm))
    ny = int(math.floor(We / block_h_mm))
    
    y_offset = (We - ny*block_h_mm) / 2.0
    
    # 5) place & extrude only blocks fully inside bar_poly
    meshes = []
    for i in range(nx):
        for j in range(ny):
            tx = i * block_w_mm
            ty = j * block_h_mm + y_offset
            footprint = affinity.translate(
                geom.box(0, 0, block_w_mm, block_h_mm),
                xoff=tx, yoff=ty
            )
            if not bar_poly.contains(footprint):
                continue
            moved = affinity.translate(block_poly, xoff=tx, yoff=ty)
            mesh  = trimesh.creation.extrude_polygon(moved, thickness, engine="triangle")
            meshes.append(mesh)

    # 6) combine & export
    combined = trimesh.util.concatenate(meshes)
    combined.export(stl_path)
    print(f" Exported by-block ASTMD638 lattice → {stl_path}")

if __name__ == "__main__":
    # load one sample from ShapeSpace.mat
    mat_path = r"C:\Users\Joory\Property-awareness-Representation-Learning\datasets\Wang\ShapeSpace.mat"

    with h5py.File(mat_path, 'r') as f:
        data   = f['ShapeSpace']            # shape (N,50,50)
        sample = data[0, :, :]              # pick index 0

    # save PNG for inspection (optional)
    img8 = ((sample - sample.min())/(sample.max()-sample.min())*255).astype(np.uint8)
    Image.fromarray(img8).save('astm_sample.png')

    # build & export
    build_astm_d638_lattice_by_block(
        img_array  = img8,
        stl_path   = 'astm_by_block.stl',
        pixel_size = 0.1,
        threshold  = 128,
        thickness  = 3.2
    )