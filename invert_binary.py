# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 01:31:32 2025

@author: joory
"""

# invert_binary.py
from PIL import Image
import numpy as np

# 1) Load the new binary from mentor’s model
img = Image.open("vae_recon_sample34_binary.png").convert("L")
arr = np.array(img, dtype=np.uint8)

# 2) Invert black ↔ white
inv = 255 - arr

# 3) Save *over* the old file your builder expects
Image.fromarray(inv).save("vae_output_sample34_binary.png")
print("Inverted and saved ▶ vae_output_sample34_binary.png")
