import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 设置全局字体
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# 读取数据
data = pd.read_csv('sampleData/sample.txt', delim_whitespace=True, header=None)

# 从第4到第16列作为特征，第17列作为标签
X = data.iloc[:, 3:16]
y = data.iloc[:, 16]

# 重新编号特征列为 0-12
X.columns = [f"feature{i}" for i in range(X.shape[1])]

# 定义与热力图一致的特征标签（含数学符号）
heatmap_labels = [
    "R", "G", "B", "H", "S", "V", r"$G_r$",
    "Linearity", "Planarity", "Scatter",
    r"$\bar{h}$", r"$\sigma_h$", r"$sk_h$"
]

# 训练随机森林模型
model = RandomForestClassifier(random_state=0)
model.fit(X, y)

# 获取特征重要性并排序
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# 绘制特征重要性直方图
plt.figure(figsize=(10, 6))
bars = plt.bar(range(X.shape[1]), importances[indices], align="center", color="skyblue")

# 统一横轴标签，并按特征重要性排序
sorted_labels = [heatmap_labels[idx] for idx in indices]
plt.xticks(range(X.shape[1]), sorted_labels, rotation=30)
plt.xlabel("Feature", fontsize=16)
plt.ylabel("Importance", fontsize=16)
plt.title("Feature Importance", fontsize=18)

# 设置横轴标签为斜体
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontstyle("italic")
    label.set_fontsize(14)

# 将 Y 轴刻度标签也可做相同处理（若需要）
for label in ax.get_yticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontsize(14)

# 在柱子上方添加特征重要性数值
for bar, importance in zip(bars, importances[indices]):
    plt.text(bar.get_x() + bar.get_width() / 2,  # X 坐标：柱子的中心
             bar.get_height() + 0.001,  # Y 坐标：柱子顶部 + 偏移
             f"{importance:.3f}",  # 显示 4 位小数
             ha='center', va='bottom', fontsize=12, fontname="Times New Roman")

plt.tight_layout()

# 保存图片
plt.savefig("rf_feature_importances.jpg", dpi=300, format='jpg')
plt.savefig("rf_feature_importances.svg", dpi=300, format='svg')

plt.show()
