import pandas as pd
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb  # 导入LightGBM

# 1. 读取数据
data = pd.read_csv('sampleData/sample.txt', delim_whitespace=True, header=None)  # 替换为你的txt文件路径

# 从第4到第16列作为特征，第17列作为标签
X = data.iloc[:, 6:16]  # 特征数据 (第4到16列)
y = data.iloc[:, 16]    # 标签数据 (第17列)

# 2. 使用LabelEncoder转换标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # 将标签从1-6转换为0-5

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 定义基模型，使用LightGBM替代RandomForestClassifier
model_lgbm = lgb.LGBMClassifier(n_estimators=5, random_state=42)  # 使用LightGBM
model_svm = SVC(probability=True, random_state=42)
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_knn = KNeighborsClassifier(n_neighbors=5)

# 4. 训练基模型并生成预测
model_lgbm.fit(X_train, y_train)
model_svm.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)
model_knn.fit(X_train, y_train)

# 生成基模型在测试集上的预测概率
pred_lgbm = model_lgbm.predict_proba(X_test)
pred_svm = model_svm.predict_proba(X_test)
pred_xgb = model_xgb.predict_proba(X_test)
pred_knn = model_knn.predict_proba(X_test)

# 将所有基模型的预测结果合并作为新的特征
stacked_features = np.hstack((pred_lgbm, pred_svm, pred_xgb, pred_knn))

# 5. 定义元模型并进行训练
meta_model = LogisticRegression()
meta_model.fit(stacked_features, y_test)  # 使用基模型的预测结果和真实标签训练元模型

# 6. 在测试集上进行最终预测
final_predictions = meta_model.predict(stacked_features)

# 7. 评估模型
accuracy = accuracy_score(y_test, final_predictions)
print(f'最终预测的准确率: {accuracy * 100:.2f}%')
