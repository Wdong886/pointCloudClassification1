import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize, StandardScaler
import pandas as pd

# 1. 加载数据
data = np.loadtxt("sampleData/sample.txt")
X = data[:, 6:16]  # 特征数据 (第7到16列，共10个特征)
y = data[:, 16]    # 标签数据 (第17列)

# 2. 划分训练集和测试集（此时不进行任何数据预处理）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 在训练集上进行标准化，防止数据泄露
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 使用交叉验证来评估模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
print(f"交叉验证AUC得分: {cv_scores}")
print(f"AUC平均值: {cv_scores.mean()}")

# 5. 在整个训练集上训练模型，然后在测试集上评估
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)  # 使用 predict_proba 获取概率分数

# 6. 输出分类报告和混淆矩阵
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型精度: {accuracy:.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred))
print("混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)

# 可视化 1：混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("混淆矩阵")
plt.show()

# 可视化 2：ROC 曲线和 AUC（适用于二分类或多分类）
# 若为多分类，将标签二值化以适配 ROC 曲线
y_bin = label_binarize(y_test, classes=np.unique(y))

plt.figure(figsize=(10, 8))
for i in range(y_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])  # 使用概率分数绘制曲线
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Category {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("假正率 (FPR)")
plt.ylabel("真正率 (TPR)")
plt.title("ROC 曲线")
plt.legend(loc="lower right")
plt.show()

# 可视化 3：Precision-Recall 曲线
plt.figure(figsize=(10, 8))
for i in range(y_bin.shape[1]):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_pred_proba[:, i])
    plt.plot(recall, precision, label=f"Category {i}")

plt.xlabel("召回率 (Recall)")
plt.ylabel("精确率 (Precision)")
plt.title("Precision-Recall 曲线")
plt.legend(loc="upper right")
plt.show()

# 可视化 4：分类报告热力图
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="YlGnBu")
plt.title("分类报告热力图")
plt.show()

# 可视化 5：学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=10, random_state=42), X, y, cv=5,
    scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, "o-", color="r", label="训练集准确率")
plt.plot(train_sizes, test_mean, "o-", color="g", label="验证集准确率")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.xlabel("训练样本数量")
plt.ylabel("准确率")
plt.title("学习曲线")
plt.legend(loc="best")
plt.grid()
plt.show()
