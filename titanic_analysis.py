import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import ttest_ind

# ======= Đọc dữ liệu =======
df = pd.read_csv('train.csv')

# ======= Xử lý thiếu Age bằng Linear Regression =======
age_df = df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch']].copy()
age_df['Sex'] = age_df['Sex'].map({'male': 0, 'female': 1})
known_age = age_df[age_df['Age'].notnull()]
unknown_age = age_df[age_df['Age'].isnull()]
X_train_age = known_age.drop('Age', axis=1)
y_train_age = known_age['Age']
lr = LinearRegression()
lr.fit(X_train_age, y_train_age)
predicted_ages = lr.predict(unknown_age.drop('Age', axis=1))
df.loc[df['Age'].isnull(), 'Age'] = predicted_ages

# ======= Xử lý thiếu Embarked bằng KNN =======
knn_df = df[['Embarked', 'Pclass', 'Sex', 'Fare']].copy()
knn_df['Sex'] = knn_df['Sex'].map({'male': 0, 'female': 1})
known_embarked = knn_df[knn_df['Embarked'].notnull()]
unknown_embarked = knn_df[knn_df['Embarked'].isnull()]
X_emb = known_embarked[['Pclass', 'Sex', 'Fare']]
y_emb = known_embarked['Embarked']
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_emb, y_emb)
predicted_embarked = knn.predict(unknown_embarked[['Pclass', 'Sex', 'Fare']])
df.loc[df['Embarked'].isnull(), 'Embarked'] = predicted_embarked

# ======= Trích xuất Deck từ Cabin =======
df['Deck'] = df['Cabin'].str[0]
df['Deck'] = df['Deck'].fillna('U')  # Unknown

# ======= Tạo đặc trưng mới =======
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
df['FarePerPerson'] = df['Fare'] / df['FamilySize']

# ======= Boxplot: Fare per person theo Pclass và Survived =======
plt.figure(figsize=(8, 5))
sns.boxplot(x='Pclass', y='FarePerPerson', hue='Survived', data=df)
plt.title('Phân phối Fare per Person theo Pclass và Survived')
plt.tight_layout()
plt.show()

# ======= Kiểm định t-test cho FarePerPerson =======
group1 = df[df['Survived'] == 1]['FarePerPerson']
group0 = df[df['Survived'] == 0]['FarePerPerson']
t_stat, p_val = ttest_ind(group1, group0)
print(f"T-test kết quả: t={t_stat:.3f}, p={p_val:.3f}")

# ======= Pipeline tiền xử lý =======
num_cols = ['Age', 'FarePerPerson', 'FamilySize']
cat_cols = ['Sex', 'Embarked', 'Deck', 'Title', 'IsAlone', 'Pclass']

num_transformer = Pipeline([
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# ======= Random Forest và Grid Search =======
X = df[num_cols + cat_cols]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, None]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

# ======= Đánh giá mô hình =======
y_pred = grid.predict(X_test)
y_proba = grid.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# ======= Cross-validation =======
y_pred_cv = cross_val_predict(grid.best_estimator_, X, y, cv=5, method='predict')
y_proba_cv = cross_val_predict(grid.best_estimator_, X, y, cv=5, method='predict_proba')[:, 1]

print("\nCross-Validation Metrics:")
print("Accuracy:", accuracy_score(y, y_pred_cv))
print("Precision:", precision_score(y, y_pred_cv))
print("Recall:", recall_score(y, y_pred_cv))
print("F1-score:", f1_score(y, y_pred_cv))
print("ROC AUC:", roc_auc_score(y, y_proba_cv))

# ======= Feature Importance =======
clf = grid.best_estimator_.named_steps['classifier']
features = grid.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
importances = clf.feature_importances_

feat_imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(15))
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()
