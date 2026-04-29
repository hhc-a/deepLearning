import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Input, Dense

# 固定亂數種子，可讓每次執行時的資料打亂結果一致，方便重現實驗結果。
np.random.seed(7)

# 載入波士頓房價資料集，並轉成 NumPy 陣列以便後續處理。
df = pd.read_csv("./boston_housing.csv")
dataset = df.values

# TODO 1. 先將資料順序打亂，可避免原始排序影響訓練與測試結果。
np.random.shuffle(dataset)

# TODO 2. 切分資料：前 13 欄為房屋特徵，最後 1 欄為房價標籤。
# for i in range(0,10,1)
X = dataset[:, 0:13] #slice --> view
y = dataset[:, 13]

# TODO 3. 前處理/正規化：將各特徵做標準化，使其平均值接近 0、標準差接近 1，
# 有助於神經網路更穩定地學習。
X -= X.mean(axis=0)
X /= X.std(axis=0)

# TODO 4. 將資料切分為訓練集與測試集。
# 其中前 404 筆作為訓練資料，後 102 筆作為測試資料。
X_train, y_train = X[:404], y[:404]
X_test, y_test = X[404:], y[404:]


# 建立深度神經網路回歸模型。
def build_deep_model():
    model = Sequential()

    # 輸入層的維度等於特徵數量，本例共有 13 個輸入特徵。
    model.add(Input(shape=(X_train.shape[1],)))

    # 第一與第二個隱藏層使用 ReLU 啟動函數，負責學習輸入特徵與房價之間的非線性關係。
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))

    # TODO 5. 輸出層只有 1 個神經元，代表預測的房價數值。
    model.add(Dense(1))

    # 編譯模型：在正式訓練前，先設定學習時要使用的損失函數、
    # 權重更新方法，以及訓練過程中要觀察的評估指標。
    # loss="mse" 表示使用平均平方誤差，適合處理房價這類回歸問題。
    # optimizer="adam" 表示使用 Adam 最佳化器來調整模型權重。
    # metrics=["mae"] 表示額外觀察平均絕對誤差，方便了解預測平均差距。
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    #
    return model


# 使用 4-fold 交叉驗證評估模型穩定性。
# 也就是將訓練資料分成 4 份，輪流取其中 1 份作驗證集，其餘作訓練集。
k = 4
nb_val_samples = len(X_train) // k
nb_epochs = 80
mse_scores = []
mae_scores = []

# 建立一個新模型。
model = build_deep_model()
#
for i in range(k):
    print("Processing Fold #" + str(i))

    # 取出目前 fold 的驗證資料。
    X_val = X_train[i*nb_val_samples: (i+1)*nb_val_samples]
    y_val = y_train[i*nb_val_samples: (i+1)*nb_val_samples]

    # 將剩餘資料組合成目前 fold 的訓練資料。
    X_train_p = np.concatenate(
            [X_train[:i*nb_val_samples],
            X_train[(i+1)*nb_val_samples:]], axis=0)
    y_train_p = np.concatenate(
            [y_train[:i*nb_val_samples],
            y_train[(i+1)*nb_val_samples:]], axis=0)

    # 課本原本做法（比較怪，故移除）：
    # 每個 fold 都重新建立一個新模型，避免前一次訓練結果影響本次實驗。
    # model = build_deep_model()

    # 使用目前 fold 的訓練資料進行模型訓練。
    model.fit(X_train_p, y_train_p, epochs=nb_epochs, 
              batch_size=16, verbose=0)

    # TODO 6. 使用驗證資料評估目前模型，並記錄 MSE 與 MAE。
    mse, mae = model.evaluate(X_val, y_val, verbose=0)
    mse_scores.append(mse)
    mae_scores.append(mae)
    

# TODO 7. 將 4 次交叉驗證的結果取平均，可作為模型在驗證階段的整體表現。
print("MSE_val: ", np.mean(mse_scores))
print("MAE_val: ", np.mean(mae_scores))

# 最後再使用測試集評估模型，檢查其對未知資料的預測能力。
mse, mae = model.evaluate(X_test, y_test, verbose=0)    
print("MSE_test: ", mse)
print("MAE_test: ", mae)