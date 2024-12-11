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

# 3. 定义基模型
model_rf = RandomForestClassifier(n_estimators=5, random_state=42)
model_svm = SVC(probability=True, random_state=42)
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_knn = KNeighborsClassifier(n_neighbors=5)

# 4. 训练基模型并生成预测
model_rf.fit(X_train, y_train)
model_svm.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)
model_knn.fit(X_train, y_train)

# 生成基模型在测试集上的预测概率
pred_rf = model_rf.predict_proba(X_test)
pred_svm = model_svm.predict_proba(X_test)
pred_xgb = model_xgb.predict_proba(X_test)
pred_knn = model_knn.predict_proba(X_test)

# 将所有基模型的预测结果合并作为新的特征
stacked_features = np.hstack((pred_rf, pred_svm, pred_xgb, pred_knn))

# 5. 定义元模型并进行训练
meta_model = LogisticRegression()
meta_model.fit(stacked_features, y_test)  # 使用基模型的预测结果和真实标签训练元模型

# 6. 在测试集上进行最终预测
final_predictions = meta_model.predict(stacked_features)

# 7. 评估模型
accuracy = accuracy_score(y_test, final_predictions)
print(f'最终预测的准确率: {accuracy * 100:.2f}%')

# 混淆矩阵
cm = confusion_matrix(y_test, final_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicting label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.show()

# ROC 曲线和 AUC（适用于多分类）
# 若为多分类，将标签二值化以适配 ROC 曲线
y_bin = label_binarize(y_test, classes=np.unique(y))
final_predictions_bin = label_binarize(final_predictions, classes=np.unique(y))

plt.figure(figsize=(10, 8))
for i in range(y_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_bin[:, i], final_predictions_bin[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Category {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
plt.savefig('result/stacking/ROC Curve.jpg', format='jpg', dpi=300)  # dpi 决定图像分辨率，通常为 300
# 保存为 EPS 格式（矢量图）
plt.savefig('result/stacking/ROC Curve.eps', format='eps')

# Precision-Recall 曲线
plt.figure(figsize=(10, 8))
for i in range(y_bin.shape[1]):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], final_predictions_bin[:, i])
    plt.plot(recall, precision, label=f"Category {i}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="upper right")
plt.show()
plt.savefig('result/stacking/Precision-Recall.jpg', format='jpg', dpi=300)  # dpi 决定图像分辨率，通常为 300
plt.savefig('result/stacking/Precision-Recall.eps', format='eps')

# 分类报告热力图
report = classification_report(y_test, final_predictions, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="YlGnBu")
plt.title("Classification Report Heatmap")
plt.show()
plt.savefig('result/stacking/Classification Report Heatmap.jpg', format='jpg', dpi=300)  # dpi 决定图像分辨率，通常为 300
plt.savefig('result/stacking/Classification Report Heatmap.eps', format='eps')

# 学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=10, random_state=42), X, y, cv=5,
    scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, "o-", color="r", label="Accuracy of the Training Set")
plt.plot(train_sizes, test_mean, "o-", color="g", label="Accuracy of the Validation Set")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.grid()
plt.show()
plt.savefig('result/stacking/Learning Curve.jpg', format='jpg', dpi=300)  # dpi 决定图像分辨率，通常为 300
plt.savefig('result/stacking/Learning Curve.eps', format='eps')