import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# 1. Đọc dữ liệu
file_path = 'Data_Number_6.csv'  # Đảm bảo file CSV nằm cùng thư mục hoặc thay bằng đường dẫn đầy đủ
df = pd.read_csv(file_path)

# 2. Tiền xử lý dữ liệu
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
density_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['traffic_density_num'] = df['traffic_density'].map(density_map)

# 3. Phân cụm điểm tắc nghẽn bằng KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['x_coord', 'y_coord', 'hour', 'traffic_density_num']])

# 4. Tính chỉ số mức độ nghiêm trọng của tắc nghẽn
df['congestion_score'] = df['traffic_density_num'] * (1 / (df['speed'] + 1))

# 5. Tạo đặc trưng giờ cao điểm
df['rush_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9 or 16 <= x <= 18) else 0)

# 6. Tạo đặc trưng tỷ lệ xe lớn (Car + Bus)
def is_large_vehicle(vtype):
    return 1 if vtype in ['Car', 'Bus'] else 0

df['is_large_vehicle'] = df['vehicle_type'].apply(is_large_vehicle)
df['rounded_x'] = df['x_coord'].round(2)
df['rounded_y'] = df['y_coord'].round(2)
ratio_df = df.groupby(['rounded_x', 'rounded_y'])['is_large_vehicle'].mean().reset_index()
ratio_df.rename(columns={'is_large_vehicle': 'large_vehicle_ratio'}, inplace=True)
df = df.merge(ratio_df, on=['rounded_x', 'rounded_y'], how='left')

# 7. Xây dựng mô hình Gradient Boosting để dự đoán mật độ giao thông
X = df[['x_coord', 'y_coord', 'hour', 'speed', 'rush_hour', 'large_vehicle_ratio']]
y = df['traffic_density']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nBáo cáo đánh giá mô hình:")
print(classification_report(y_test, y_pred))