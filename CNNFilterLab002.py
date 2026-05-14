import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# TODO 1. 挑選數字
img = np.load("digit4.npy")
filters = [[
    [-1, -1, -1],
    [ 1,  1,  1],
    [ 0,  0,  0]],
   [[-1,  1,  0],
    [-1,  1,  0],
    [-1,  1,  0]],
   [[ 0,  0,  0],
    [ 1,  1,  1],
    [-1, -1, -1]],
   [[ 0,  1, -1],
    [ 0,  1, -1],
    [ 0,  1, -1]]]

#
plt.figure()
plt.subplot(1, 5, 1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("original")
#
for i in range(2, 6):
    # TODO 2. 畫圖
    plt.subplot(1, 5, i)
    # TODO 3. 濾鏡處理
    c = signal.convolve2d(img, filters[i-2], boundary="symm", mode="same")
    plt.imshow(c, cmap="gray")
    #
    plt.axis("off")
    plt.title("filter"+str(i-1))
#
plt.show()