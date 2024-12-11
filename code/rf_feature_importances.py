import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 读取数据
data = pd.read_csv('sampleData/sample.txt', delim_whitespace=True, header=None)  # 替换为你的txt文件路径

# 从第4到第16列作为特征，第17列作为标签
X = data.iloc[:, 3:16]  # 特征数据 (第4到16列)
y = data.iloc[:, 16]  # 标签数据 (第17列)

# 重新编号特征列为0-12
X.columns = [f"feature{i}" for i in range(X.shape[1])]

# 训练随机森林模型
model = RandomForestClassifier(random_state=0)
model.fit(X, y)

# 获取特征重要性并排序
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # 按重要性排序

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices], align="center", color="skyblue")
plt.xticks(range(X.shape[1]), [f"feature{i}" for i in indices], rotation=45, fontsize=10, fontname="Times New Roman")
plt.yticks(fontsize=10, fontname="Times New Roman")
plt.xlabel("Feature Index", fontsize=12, fontname="Times New Roman")
plt.ylabel("Importance", fontsize=12, fontname="Times New Roman")
plt.title("Feature Importance using Random Forest", fontsize=14, fontname="Times New Roman")
plt.tight_layout()

plt.savefig(r"C:\Users\wdong\Desktop\rf_feature_importances.jpg", dpi=300, format='jpg')
plt.savefig(r"C:\Users\wdong\Desktop\rf_feature_importances.eps", format='eps')
# 显示图形
plt.show()
