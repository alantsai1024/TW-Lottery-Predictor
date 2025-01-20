import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Attention, Input
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping

# 在開頭添加以下設置
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 讀取數據
def load_data(file_path):
    df = pd.read_excel('data.xlsx')
     
    return df

# 數據預處理
def preprocess_data(df):
    # 將日期轉換為時間特徵
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # 對號碼進行歸一化
    scaler = MinMaxScaler()
    numbers = df[['num1', 'num2', 'num3', 'num4', 'num5']].values
    normalized_numbers = scaler.fit_transform(numbers)
    
    return normalized_numbers, scaler

# 創建序列數據
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# 建立Attention LSTM模型
def create_model(seq_length, n_features):
    inputs = Input(shape=(seq_length, n_features))
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    attention_layer = Attention()([lstm_out, lstm_out])
    lstm_out2 = LSTM(32)(attention_layer)
    dense1 = Dense(32, activation='relu')(lstm_out2)
    outputs = Dense(n_features)(dense1)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# 主函數
def predict_next_numbers():
    # 讀取數據
    df = load_data('data.xlsx')
    
    # 預處理數據
    normalized_numbers, scaler = preprocess_data(df)
    
    # 創建序列
    seq_length = 10  # 使用過去10期數據預測
    X, y = create_sequences(normalized_numbers, seq_length)
    
    # 分割訓練集和測試集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 創建和訓練模型
    model = create_model(seq_length, 5)
    
    # 添加 EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,  # 如果20個epoch內驗證損失沒有改善，就停止訓練
        min_delta=0.0001,  # 改善必須大於0.0001才算數
        restore_best_weights=True
    )
    
    # 修改 model.fit
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 預測下一期號碼
    last_sequence = normalized_numbers[-seq_length:]
    predicted_normalized = model.predict(last_sequence.reshape(1, seq_length, 5))
    predicted_numbers = scaler.inverse_transform(predicted_normalized)
    
    # 將預測結果轉換為整數
    predicted_numbers = np.round(predicted_numbers[0]).astype(int)
    
    # 確保預測的號碼在1-39範圍內
    predicted_numbers = np.clip(predicted_numbers, 1, 39)
    
    # 排序預測結果
    predicted_numbers.sort()
    
    return predicted_numbers, history

# 執行預測
if __name__ == "__main__":
    predicted_numbers, history = predict_next_numbers()
    print(f"預測下一期號碼：{predicted_numbers}")
    
    # 繪製損失函數圖表
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='訓練損失')
    plt.plot(history.history['val_loss'], label='驗證損失')
    plt.title('模型訓練損失')
    plt.xlabel('訓練週期')
    plt.ylabel('損失值')
    plt.legend()
    plt.show()
