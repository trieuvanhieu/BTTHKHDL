import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import seaborn as sns

# Load dữ liệu
df = pd.read_csv("Data_Number_4.csv")

# Chuyển timestamp sang datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Phân cụm DBSCAN
features = df[['x_coord', 'y_coord', 'pm25']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

dbscan = DBSCAN(eps=0.5, min_samples=3)
df['cluster'] = dbscan.fit_predict(scaled)

# Vẽ phân cụm
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='x_coord', y='y_coord', hue='cluster', palette='tab10')
plt.title("Phân cụm DBSCAN theo vị trí và PM2.5")
plt.show()

# 2. Tính chỉ số rủi ro ô nhiễm
def calc_risk(gr):
    pm25_avg = gr['pm25'].mean()
    exceed_count = (gr['pm25'] > 50).sum()
    total = len(gr)
    return pd.Series({
        'pm25_mean': pm25_avg,
        'exceed_freq': exceed_count / total,
        'risk_index': pm25_avg * (exceed_count / total)
    })

risk_df = df.groupby('cluster').apply(calc_risk).reset_index()
print(risk_df)

# 3. Chỉ số thời tiết bất lợi (tự định nghĩa)
# Độ ẩm cao (>80) + Gió yếu (<5) => +1 điểm
def adverse_weather(row):
    score = 0
    if row['humidity'] > 80: score += 1
    if row['wind_speed'] < 5: score += 1
    return score

df['adverse_weather_index'] = df.apply(adverse_weather, axis=1)

# 4. Xu hướng ô nhiễm: Độ dốc PM2.5 theo thời gian (rolling slope)
df = df.sort_values(['x_coord', 'y_coord', 'timestamp'])
df['pm25_trend'] = df.groupby(['x_coord', 'y_coord'])['pm25'].transform(
    lambda x: x.rolling(24, min_periods=1).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0])
)

# 5. Dự đoán bằng LSTM
# Chọn 1 vị trí cụ thể
position = df[['x_coord', 'y_coord']].drop_duplicates().iloc[0]
df_pos = df[(df['x_coord'] == position['x_coord']) & (df['y_coord'] == position['y_coord'])].copy()
df_pos = df_pos.sort_values('timestamp')

# Chuẩn bị dữ liệu LSTM
def create_lstm_data(series, input_len=24, pred_len=6):
    X, y = [], []
    for i in range(len(series) - input_len - pred_len + 1):
        X.append(series[i:i+input_len])
        y.append(series[i+input_len:i+input_len+pred_len])
    return np.array(X), np.array(y)

pm_series = df_pos['pm25'].values
X, y = create_lstm_data(pm_series)

# Reshape cho LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Mô hình
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(6)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=0)

# Dự đoán
y_pred = model.predict(X)

# Đánh giá RMSE
rmse = sqrt(mean_squared_error(y.flatten(), y_pred.flatten()))
print(f"RMSE: {rmse:.2f}")

# Vẽ biểu đồ thực tế vs dự đoán
plt.figure(figsize=(10, 6))
plt.plot(y.flatten()[:100], label="Thực tế")
plt.plot(y_pred.flatten()[:100], label="Dự đoán")
plt.legend()
plt.title("So sánh PM2.5 thực tế và dự đoán (vị trí đầu tiên)")
plt.show()
