import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

img = np.load("digit4.npy")
#
plt.figure(figsize=(10,8))
# 原圖
plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Original image")

# define the edge filter kernel
kernel = [
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
    ]
    
# Filter - edge
# TODO 1. Filter
filtered_img = signal.convolve2d(img, kernel, boundary="symm", mode="same")

# TODO 2. show the result 
plt.subplot(1,2,2)
plt.imshow(filtered_img, cmap="gray")
plt.axis("off")
plt.title("Edge-filtered image")
#
plt.show()
   