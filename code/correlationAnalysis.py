import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据
data = pd.read_csv('sampleData/sample.txt', delim_whitespace=True, header=None)  # 替换为你的txt文件路径

# 假设数据列结构为：第1到3列为坐标，不作为特征，4到16列为特征，17列为标签
X = data.iloc[:, 3:16]  # 选择特征列（第4到16列）

# 2. 计算相关性矩阵
corr_matrix = X.corr()

# 3. 可视化相关性矩阵
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()

# 4. 筛选相关性较高的特征（如果需要）
threshold = 0.9  # 设置相关性阈值
high_corr_var = np.where(corr_matrix > threshold)

# 找到相关性大于阈值的特征对（去除对称部分，避免重复）
high_corr_var = [(X.columns[x], X.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y]

# 输出高度相关的特征对
print("Highly correlated feature pairs (correlation > 0.9):")
print(high_corr_var)

# 如果需要根据相关性删除特征，你可以在此处进行操作，例如删除高相关的特征
