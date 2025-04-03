import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
from sklearn.metrics import f1_score, matthews_corrcoef

# 1. 读取数据
data = pd.read_csv('../sampleData/sample.txt', delim_whitespace=True, header=None)  # 替换为你的txt文件路径

# 从第4到第16列作为特征，第17列作为标签
X = data.iloc[:, 9:16]  # 特征数据 (第4到16列)
y = data.iloc[:, 16]    # 标签数据 (第17列)

# 2. 使用LabelEncoder转换标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # 将标签从1-6转换为0-5

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 3. 定义基模型（删除KNN，仅保留RF、SVM、XGBoost）
model_rf = RandomForestClassifier(n_estimators=5, random_state=42)
model_svm = SVC(probability=True, random_state=42)
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# 4. 训练基模型并生成预测（移除KNN的训练）
model_rf.fit(X_train, y_train)
model_svm.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)

# 生成基模型在测试集上的预测概率（删除KNN的预测）
pred_rf = model_rf.predict_proba(X_test)
pred_svm = model_svm.predict_proba(X_test)
pred_xgb = model_xgb.predict_proba(X_test)

# 合并三个基模型的预测结果（移除pred_knn）
stacked_features = np.hstack((pred_rf, pred_svm, pred_xgb))  # 仅3个基模型的概率堆叠

# 5. 定义元模型并进行训练
meta_model = LogisticRegression()
meta_model.fit(stacked_features, y_test)  # 使用基模型的预测结果和真实标签训练元模型

# 6. 在测试集上进行最终预测
final_predictions = meta_model.predict(stacked_features)

# 7. 评估模型
accuracy = accuracy_score(y_test, final_predictions)
f1 = f1_score(y_test, final_predictions, average='macro')  # 使用宏平均计算多分类F1
mcc = matthews_corrcoef(y_test, final_predictions)         # 直接支持多分类

print(f"最终预测的准确率: {accuracy * 100:.2f}%")
print(f"F1分数（宏平均）: {f1:.4f}")
print(f"马修斯相关系数（MCC）: {mcc:.4f}")