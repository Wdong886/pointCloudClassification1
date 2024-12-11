import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# 读取文件并清理每行数据格式
cleaned_lines = []
with open("sampleData/sample.txt", "r") as f:
    for line in f:
        # 去掉每行末尾的多余空格，并确保字段数一致
        parts = line.strip().split()
        if len(parts) == 17:
            cleaned_lines.append(" ".join(parts))
        else:
            # 如果行字段数不一致，可以选择忽略或修正此行
            print(f"Line with unexpected field count: {line.strip()}")

# 将清理后的数据写入新文件
with open("sampleData/sample_clean.txt", "w") as f:
    f.write("\n".join(cleaned_lines))

print("Data cleaned and saved to sample_clean.txt")

# 加载数据
data = pd.read_csv("sampleData/sample_clean.txt", delimiter=' ', header=None)
X = data.iloc[:, 3:16]  # 第4到16列是特征
y = data.iloc[:, 16]    # 第17列是标签

# 步骤1：计算特征相关性矩阵并可视化
correlation_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# 设置相关性阈值
correlation_threshold = 0.8  # 选择相关系数较高的特征对

# 步骤2：训练随机森林并计算特征重要性
clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(X, y)
feature_importances = clf.feature_importances_

# 打印每个特征的重要性
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
print("随机森林特征重要性:\n", feature_importance_df.sort_values(by="Importance", ascending=False))

# 步骤3：找到相关性较高的特征对并筛选重要特征
# 获取高相关性特征对的索引
high_corr_pairs = np.where(np.abs(correlation_matrix) > correlation_threshold)
high_corr_pairs = [(i, j) for i, j in zip(*high_corr_pairs) if i < j]

# 筛选相关性高的特征对并比较重要性
selected_features = set(range(X.shape[1]))  # 初始保留所有特征
print("筛选前的特征索引:", selected_features)
for i, j in high_corr_pairs:
    # 比较特征i和特征j的重要性，保留重要性较高的特征
    if feature_importances[i] >= feature_importances[j]:
        selected_features.discard(j)
    else:
        selected_features.discard(i)

# 最终筛选后的特征索引
selected_features = list(selected_features)
print("筛选后的特征索引:", selected_features)

# 使用筛选后的特征重新创建数据集
X_selected = X.iloc[:, selected_features]
print("筛选后的特征数据集:\n", X_selected.head())

# （可选）步骤4：测试筛选效果 - 使用筛选后的特征重新训练模型并评估效果
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"筛选后模型精度: {accuracy:.4f}")
