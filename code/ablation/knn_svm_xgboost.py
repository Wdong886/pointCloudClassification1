import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report

# 1. 读取数据
data = pd.read_csv('../sampleData/sample.txt', delim_whitespace=True, header=None)

# 特征和标签提取
X = data.iloc[:, 9:16]  # 特征数据
y = data.iloc[:, -1]    # 标签数据

# 2. 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 3. 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 定义基模型：KNN, SVM, XGBoost
print("训练基模型中...")
model_knn = KNeighborsClassifier(n_neighbors=5)
model_svm = SVC(probability=True, random_state=42)
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# 5. 训练基模型
model_knn.fit(X_train, y_train)
model_svm.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)

# 6. 生成基模型的预测概率
print("生成基模型预测...")
pred_knn = model_knn.predict_proba(X_test)
pred_svm = model_svm.predict_proba(X_test)
pred_xgb = model_xgb.predict_proba(X_test)

# 7. 堆叠基模型的预测结果
stacked_features = np.hstack((pred_knn, pred_svm, pred_xgb))

# 8. 定义并训练元模型（Logistic Regression）
print("训练元模型...")
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(stacked_features, y_test)

# 9. 最终预测
final_predictions = meta_model.predict(stacked_features)

# 10. 评估模型
accuracy = accuracy_score(y_test, final_predictions)
f1 = f1_score(y_test, final_predictions, average='macro')
mcc = matthews_corrcoef(y_test, final_predictions)

print("\n=== 模型评估结果 ===")
print(f"准确率: {accuracy * 100:.2f}%")
print(f"F1分数（宏平均）: {f1:.4f}")
print(f"马修斯相关系数（MCC）: {mcc:.4f}")

# 11. 输出分类报告
print("\n=== 分类报告 ===")
print(classification_report(y_test, final_predictions, target_names=label_encoder.classes_))