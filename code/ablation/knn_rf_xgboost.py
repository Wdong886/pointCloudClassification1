import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier

# 1. 读取数据
data = pd.read_csv('../sampleData/sample.txt', delim_whitespace=True, header=None)

# 特征和标签提取
X = data.iloc[:, 9:16]
y = data.iloc[:, -1]

# 2. 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 定义基模型
model_knn = KNeighborsClassifier(n_neighbors=5)
model_rf = RandomForestClassifier(n_estimators=5, random_state=42)
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)


# 4. 训练基模型
print("训练基模型中...")
model_knn.fit(X_train, y_train)
model_rf.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)

# 生成基模型的预测概率
print("生成基模型预测...")
pred_knn = model_knn.predict_proba(X_test)
pred_rf = model_rf.predict_proba(X_test)
pred_xgb = model_xgb.predict_proba(X_test)

# 合并三个基模型的预测结果
stacked_features = np.hstack((pred_knn, pred_rf,pred_xgb))

# 5. 定义并训练元模型
print("训练元模型...")
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(stacked_features, y_test)

# 6. 在测试集上进行最终预测
print("进行最终预测...")
final_predictions = meta_model.predict(stacked_features)

# 7. 评估模型
accuracy = accuracy_score(y_test, final_predictions)
f1 = f1_score(y_test, final_predictions, average='macro')
mcc = matthews_corrcoef(y_test, final_predictions)

print("\n模型评估结果:")
print(f"准确率: {accuracy * 100:.2f}%")
print(f"F1分数(宏平均): {f1:.4f}")
print(f"马修斯相关系数(MCC): {mcc:.4f}")

# 8. 输出分类报告
from sklearn.metrics import classification_report
print("\n分类报告:")
print(classification_report(y_test, final_predictions, target_names=label_encoder.classes_))