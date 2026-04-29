import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#
import pandas as pd

# 載入資料集
df = pd.read_csv("./iris_data.csv")
# 查看前5筆記錄
print(df.head())
df.head().to_html("ch6-1-1a_01.html")
print("--------------------------")
# 顯示資料集的描述資料
print(df.describe())
df.describe().to_html("ch6-1-1a_02.html")


# 載入資料集
df = pd.read_csv("./iris_data.csv")

# TODO 1. 文字資料 --> 數值
# 建立類別名稱與數值標籤的對應表，方便後續視覺化與模型訓練使用。
target_mapping = {"setosa": 0, "versicolor": 1, "virginica": 2}
# 將 target 欄位中的文字類別轉換為數值類別。
y = df["target"].map(target_mapping)

# 使用 Matplotlib 畫出兩組散佈圖，觀察不同鳶尾花種類在特徵空間中的分布情形。
# `colmap` 中的三種顏色分別對應三個類別標籤 0、1、2。
colmap = np.array(["r", "g", "y"])
plt.figure(figsize=(10,5))

# 第一張圖比較萼片長度（sepal_length）與萼片寬度（sepal_width）的分布。
plt.subplot(1, 2, 1)
plt.subplots_adjust(hspace = .5)
# TODO 2. 散佈圖
plt.scatter(df["sepal_length"], df["sepal_width"], color=colmap[y])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

# 第二張圖比較花瓣長度（petal_length）與花瓣寬度（petal_width）的分布。
plt.subplot(1, 2, 2)
# TODO 3. 散佈圖
plt.scatter(df["petal_length"], df["petal_width"], color=colmap[y])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

# 顯示兩張散佈圖，便於觀察不同類別是否容易被區分。
plt.show()

import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

# 使用 Seaborn 的 pairplot 一次檢視多組特徵兩兩之間的分布關係。
sns.pairplot(df, hue="target")