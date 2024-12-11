import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# 1. 加载数据
data = np.loadtxt("sampleData/sample.txt")
X = data[:, [3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]]  # 选择特定特征列
y = data[:, 16]  # 标签数据 (第17列)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建并训练支持向量机模型
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_clf.fit(X_train, y_train)

# 4. 预测
y_pred = svm_clf.predict(X_test)

# 5. 计算并输出评价指标
accuracy = accuracy_score(y_test, y_pred)
print(f"模型精度: {accuracy:.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred))
print("混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)

# 可视化 1：混淆矩阵热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.show()

# 准备数据用于绘制 ROC 曲线和 Precision-Recall 曲线
y_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_bin.shape[1]
classifier = OneVsRestClassifier(SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
classifier.fit(X_train, label_binarize(y_train, classes=np.unique(y)))
y_score = classifier.decision_function(X_test)

# 可视化 2：ROC 曲线
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Category {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# 可视化 3：Precision-Recall 曲线
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, label=f"Category {i}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="upper right")
plt.show()

# 可视化 4：分类报告热力图
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="YlGnBu")
plt.title("Classification Report Heatmap")
plt.show()

# 可视化 5：学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    SVC(kernel='rbf', C=1.0, gamma='scale'), X, y, cv=5,
    scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, "o-", color="r", label="Training accuracy")
plt.plot(train_sizes, test_mean, "o-", color="g", label="Validation accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.grid()
plt.show()
