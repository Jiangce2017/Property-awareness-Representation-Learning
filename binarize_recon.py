# -*- coding: utf-8 -*-

# crop_and_binarize.py

# crop_and_binarize.py


from PIL import Image
import numpy as np
import cv2

# 1) Load reconstrcuted image
full = Image.open("vae_recon_sample34_2.png")

# 1) Load the full 4-panel figure
# w, h = full.size
# panel_w = w // 4  # each panel is 1/4th of width

# # 2) Crop out just the 3rd panel (Reconstructed Image),
# #    skipping the top header text (adjust text_height as needed).
# text_height = 20  # pixels; raise/lower this if text still shows
# recon = full.crop((2 * panel_w, text_height, 3 * panel_w, h))
# recon.save("vae_recon_crop.png")  # for your inspection

# 3) Convert to grayscale array and run Otsu’s threshold
# gray = recon.convert("L")

# 2) Convert to grayscale array and run Otsu’s
gray = full.convert("L")
arr  = np.array(gray)
_, binary = cv2.threshold(arr, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 3) Downsample to 50×50 with nearest‐neighbor
binary = cv2.resize(
        binary,
        (50, 50),
        interpolation=cv2.INTER_NEAREST
    )

# 4) Save the mask in the format your bar-generator expects
Image.fromarray(binary).save("vae_recon_sample34_binary.png")
print("✅ Cropped & binarized → vae_recon_sample34_binary.png")

