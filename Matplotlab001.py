import numpy as np
import matplotlib.pyplot as plt
'''
x = np.linspace(-2, 2, 32)  # 建立-2到2的等距數值
print(x)                    #印出x的內容
y= x ** 2                   # 每個 x 元素平方
print(y)                    #印出y的內容
fig = plt.figure(figsize=(8,6))    # 建立畫布物件fig
ax = fig.add_subplot()             # 加入一個子圖
ax.plot(x, y, 'ro--')                 # 繪製紅色曲線圖
ax.set_title('y = x^2')           # 設定圖表標題
ax.set_title('y = sin(x)')           # 設定圖表標題
ax.set_xlabel('x')                # 設定x軸標籤
ax.set_ylabel('y')                # 設定y軸標籤
plt.show()                         # 顯示圖形
'''
x = np.linspace(0, 2*np.pi, 32)  # 建立0到2π的等距數值
y_sin = np.sin(x)                # 每個 x 元素的正弦值
y_cos = np.cos(x)                # 每個 x 元素的餘弦值
y_tan = np.tan(x)                # 每個 x 元素的正切值

fig,axs = plt.subplots(2, 2, figsize=(8,8))   # 建立2*2子圖物圖形大小為8*8
#ax=fig.add_subplot()                         # 加入第一個子圖
axs[0,0].plot(x, y_sin, 'ro--')               # 繪製紅色曲線圖
axs[0,0].set_title('y = sin(x)')              # 設定子圖(0,1)的標題
axs[0,0].set_xlabel('x')                      # 設定子圖(1,0)的標題
axs[0,0].set_ylabel('y')                      # 設定子圖(0,0)的標題
axs[0,1].plot(x, y_cos, 'ro--')               # 繪製紅色曲線圖
axs[0,1].set_title('y = cos(x)')              # 設定子圖(0,1)的標題
axs[0,1].set_xlabel('x')                      # 設定子圖(1,0)的標題
axs[0,1].set_ylabel('y')                      # 設定子圖(0,0)的標題
axs[1,0].plot(x, y_tan, 'ro--')               # 繪製紅色曲線圖
axs[1,0].set_title('y = tan(x)')              # 設定子圖(0,1)的標題
axs[1,0].set_xlabel('x')                      # 設定子圖(1,0)的標題
axs[1,0].set_ylabel('y')                      # 設定子圖(0,0)的標題

axs[1,1].scatter(x, y_sin, marker='x', color='blue')  # 繪製散點圖
axs[1,1].set_title('y = sin(x) ')             # 設定子圖(1,1)的標題
plt.show()                                    # 顯示圖形
