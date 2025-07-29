# -*- coding: utf-8 -*-

# crop_and_binarize.py

from PIL import Image
import numpy as np
import cv2

# 1) Load the full 4-panel figure
full = Image.open("vae_recon.png")
w, h = full.size
panel_w = w // 4  # each panel is 1/4th of width

# 2) Crop out just the 3rd panel (Reconstructed Image),
#    skipping the top header text (adjust text_height as needed).
text_height = 20  # pixels; raise/lower this if text still shows
recon = full.crop((2 * panel_w, text_height, 3 * panel_w, h))
recon.save("vae_recon_crop.png")  # for your inspection

# 3) Convert to grayscale array and run Otsu’s threshold
gray = recon.convert("L")
arr  = np.array(gray)
_, binary = cv2.threshold(arr, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 3b) Downsample to 50×50 with nearest‐neighbor
binary = cv2.resize(
       binary,
       (50, 50),
       interpolation=cv2.INTER_NEAREST
   )

# 4) Save the mask in the format your bar-generator expects
Image.fromarray(binary).save("vae_recon_binary.png")
print("✅ Cropped & binarized → vae_recon_binary.png")

# from PIL import Image
# import numpy as np
# import cv2

# # 1) Load the full 4-panel figure
# full = Image.open("vae_recon.png")
# w, h = full.size

# # 2) Compute panel width & crop the 3rd panel (0-indexed: panels 0,1,2,3)
# panel_w = w // 4
# # x-range of the 3rd panel:
# x0 = panel_w * 2
# x1 = panel_w * 3

# recon = full.crop((x0, 0, x1, h)).convert("L")
# recon_arr = np.array(recon)  # should be exactly 50×50

# # 3) Binarize with Otsu
# _, binary = cv2.threshold(
#     recon_arr,
#     0, 255,
#     cv2.THRESH_BINARY + cv2.THRESH_OTSU
# )

# # 4) Save
# Image.fromarray(binary).save("vae_recon_binary.png")
# print(" Cropped & binarized → vae_recon_binary.png")


# # 1) Load your saved reconstruction
# img = Image.open("vae_recon.png").convert("L")
# arr = np.array(img)              # uint8 0–255

# # 2) Compute an automatic threshold via Otsu
# _, binary = cv2.threshold(
#     arr,
#     0,                # ignored by Otsu
#     255,              # output “on” value
#     cv2.THRESH_BINARY + cv2.THRESH_OTSU
# )

# # 3) Save out the binary image
# Image.fromarray(binary).save("vae_recon_binary.png")
# print(" Saved binary image as vae_recon_binary.png")
