import numpy as np
import matplotlib.pyplot as plt

def L(w1, w2):
    return w1**2 + w2**2

def dL(w):
    return np.array([2*w[0], 2*w[1]])

def gradient_descent(w_start, df, lr, epochs):
    w1_gd = []
    w2_gd = []
    w1_gd.append(w_start[0])
    w2_gd.append(w_start[1]) 
    pre_w = w_start

    for i in range(epochs):
        w = pre_w - lr*df(pre_w)
        w1_gd.append(w[0])
        w2_gd.append(w[1])
        pre_w = w

    return np.array(w1_gd), np.array(w2_gd)

w0 = np.array([2, 4])
lr = 0.1
epochs = 40

x1 = np.arange(-5, 5, 0.05)
x2 = np.arange(-5, 5, 0.05)

w1, w2 = np.meshgrid(x1, x2)

fig1, ax1 = plt.subplots()
ax1.contour(w1, w2, L(w1, w2), levels=np.logspace(-3, 3, 30), cmap='jet')
min_point = np.array([0., 0.])
min_point_ = min_point[:, np.newaxis]
ax1.plot(*min_point_, L(*min_point_), 'r*', markersize=10)
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')

w1_gd, w2_gd = gradient_descent(w0, dL, lr, epochs)
w_gd = np.column_stack([w1_gd, w2_gd])
print(w_gd)

ax1.plot(w1_gd, w2_gd, 'bo')
for i in range(1, epochs+1):
    ax1.annotate('', xy=(w1_gd[i], w2_gd[i]), 
                   xytext=(w1_gd[i-1], w2_gd[i-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')
plt.show()