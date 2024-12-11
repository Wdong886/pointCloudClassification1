import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据（假设数据是空格分隔的）
data = pd.read_csv('sampleData/sample.txt', delim_whitespace=True, header=None)

# 假设数据列结构为：第1到3列为坐标，不作为特征，4到16列为特征，17列为标签
X = data.iloc[:, 6:16]  # 选择第4到第16列为特征
y = data.iloc[:, 16]    # 选择第17列为标签

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集，80%训练集，20%测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练KNN分类器
k = 5  # K值可以调节
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'KNN分类器的准确率: {accuracy * 100:.2f}%')

# 输出其他评估指标
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

