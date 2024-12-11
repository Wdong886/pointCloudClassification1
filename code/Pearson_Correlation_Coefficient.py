import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = pd.read_csv('sampleData/sample.txt', delim_whitespace=True, header=None)  # 替换为你的txt文件路径

# 从第4到第16列作为特征，第17列作为标签
X = data.iloc[:, 3:16]  # 特征数据 (第4到16列)
y = data.iloc[:, 16]  # 标签数据 (第17列)

# 定义特征名称
feature_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11","12"]
X.columns = feature_labels  # 设置特征名称为列标签

# 计算皮尔逊相关系数矩阵
correlation_matrix = X.corr().round(2)

# 创建图形和热力图
plt.figure(figsize=(10, 6))
sns.set(font="Times New Roman")  # 设置全局字体

# 绘制热力图
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,  # 显示相关系数数值
    cmap="coolwarm",  # 颜色图方案
    annot_kws={"size": 10, "fontstyle": "normal"},  # 数值字体设置
    fmt=".2f",  # 两位小数
    cbar_kws={"label": "Pearson Correlation Coefficient"}  # 颜色条标签
)

# 设置坐标轴字体及标签
heatmap.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=10, fontname="Times New Roman")
heatmap.set_yticklabels(feature_labels, rotation=0, fontsize=10, fontname="Times New Roman")

# 设置图标题
plt.title("Feature Correlation Heatmap", fontsize=12, fontname="Times New Roman")

# 在图下方添加图例
legend_text = "; ".join([f"{label}" for label in feature_labels])
plt.figtext(0.5, -0.1, f"Legend: {legend_text}", ha="center", fontsize=10, fontname="Times New Roman")

# 调整图布局
plt.tight_layout()
# 保存为300dpi的jpg和eps文件
plt.savefig(r"C:\Users\wdong\Desktop\correlation_heatmap.jpg", dpi=300, format='jpg')
plt.savefig(r"C:\Users\wdong\Desktop\correlation_heatmap.eps", format='eps')
plt.show()
