import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# -----------------------------
# 1. 设置字体渲染为 Times New Roman（含数学符号）
# -----------------------------
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# -----------------------------
# 2. 读取数据并定义特征标签
# -----------------------------
data = pd.read_csv('sampleData/sample.txt', delim_whitespace=True, header=None)
X = data.iloc[:, 3:16]
y = data.iloc[:, 16]

feature_labels = [
    "R", "G", "B", "H", "S", "V", r"$G_r$",
    "Linearity", "Planarity", "Scatter",
    r"$\bar{h}$", r"$\sigma_h$", r"$sk_h$"
]
X.columns = feature_labels

# -----------------------------
# 3. 计算皮尔逊相关系数并绘制热力图
# -----------------------------
corr_matrix = X.corr().round(2)
plt.figure(figsize=(10, 6))

sns.set(font="Times New Roman")
heatmap = sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    annot_kws={"size": 15, "fontstyle": "normal"},
    fmt=".2f",
    cbar_kws={"label": "Pearson Correlation Coefficient"}
)

# X 轴标签
heatmap.set_xticklabels(feature_labels, rotation=45, ha='right')

# Y 轴标签：这里将纵轴标签旋转 45 度
heatmap.set_yticklabels(feature_labels, rotation=30, va='center')

# 调整刻度标签的字号、字体样式
for label in heatmap.get_xticklabels():
    label.set_fontsize(14)
    label.set_fontstyle("italic")

for label in heatmap.get_yticklabels():
    label.set_fontsize(14)
    label.set_fontstyle("italic")

plt.title("Feature Correlation Heatmap", fontsize=18)
plt.tight_layout()

# 5. 手动获取 colorbar 并设置字体大小
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)                      # 调整 colorbar 刻度字体
cbar.set_label("Pearson Correlation Coefficient",      # 调整 colorbar 标签
               fontsize=16)


# -----------------------------
# 4. 保存图像为 300 dpi 的 JPG 与矢量图
# -----------------------------
plt.savefig("heatmap.jpg", dpi=300, format="jpg")   # 300 dpi 的 JPG
plt.savefig("heatmap.svg", format="eps")           # 矢量图 EPS（也可使用 SVG、PDF 等）
plt.savefig("heatmap.svg", format="svg")         # 若需要 SVG 格式，去掉注释即可

plt.show()
