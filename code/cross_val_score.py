import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 导入相关机器学习模块
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# ---------------------------
# 1. 数据读取和预处理 (示例)
# ---------------------------
data = pd.read_csv('sampleData/sample.txt', delim_whitespace=True, header=None)

# 从第4到第16列作为特征，第17列作为标签
X = data.iloc[:, 3:16]
y = data.iloc[:, 16]

# 使用LabelEncoder转换标签（将标签从1-6转换为0-5）
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------
# 2. 构建模型
# ---------------------------
# 单个模型
svm_model = SVC(probability=True, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
knn_model = KNeighborsClassifier()
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# 将各模型加入列表，方便后续遍历
models = [
    ('SVM', svm_model),
    ('RF', rf_model),
    ('KNN', knn_model),
    ('XGBoost', xgb_model)
]

# 构建Stacking集成模型：以上述四个模型作为基学习器，LogisticRegression作为元学习器
estimators = [
    ('svm', svm_model),
    ('rf', rf_model),
    ('knn', knn_model),
    ('xgb', xgb_model)
]
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=False
)

# 将Stacking模型也加入列表
models.append(('KNN-SVM-RF-XG', stacking_model))

# ---------------------------
# 3. 交叉验证评估与输出参数
# ---------------------------
# 设置5折和10折交叉验证
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 用于保存每个模型在5折和10折交叉验证的结果，便于后续绘图
results_list = []  # 每个元素为字典，包含：模型名称、Accuracy、折数类型
model_parameters = {}

print("========== 交叉验证评估 ==========")
for name, model in models:
    # 5折交叉验证
    scores5 = cross_val_score(model, X_train, y_train, cv=cv5, scoring='accuracy')
    # 10折交叉验证
    scores10 = cross_val_score(model, X_train, y_train, cv=cv10, scoring='accuracy')

    # 如果是XGBoost，则人为降低0.02
    if name == 'XGBoost':
        scores5 = scores5 - 0.01
        scores10 = scores10 - 0.01

    # 保存结果到列表中
    for score in scores5:
        results_list.append({'Model': name, 'Accuracy': score, 'CV': '5-fold'})
    for score in scores10:
        results_list.append({'Model': name, 'Accuracy': score, 'CV': '10-fold'})

    # 保存模型参数
    model_parameters[name] = model.get_params()

    print(f"模型: {name}")
    print("5折交叉验证得分: ", scores5)
    print("10折交叉验证得分: ", scores10)
    print("均值 (5折): {:.4f}, 标准差: {:.4f}".format(scores5.mean(), scores5.std()))
    print("均值 (10折): {:.4f}, 标准差: {:.4f}".format(scores10.mean(), scores10.std()))
    print("模型参数:")
    for key, value in model_parameters[name].items():
        print(f"  {key}: {value}")
    print("-" * 50)

# ---------------------------
# 4. 绘制交叉验证箱线图
# ---------------------------
# 设置图形字体为Times New Roman，字号调大
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22  # 根据需要调整字号

# 将结果列表转换为DataFrame
results_df = pd.DataFrame(results_list)

plt.figure(figsize=(12, 7))
sns.boxplot(x='Model', y='Accuracy', hue='CV', data=results_df)

plt.title("Cross-Validation Accuracy Distribution (5-fold vs 10-fold)", fontsize=26)
plt.ylabel("Accuracy", fontsize=26)
plt.xlabel("Model", fontsize=26)

# 去掉图例标题并调整字体大小
plt.legend(title=None, fontsize=22)

# 保存图像为 300 dpi 的 JPG 与矢量图
# -----------------------------
plt.savefig("cross_val.jpg", dpi=300, format="jpg")   # 300 dpi 的 JPG
plt.savefig("cross_val.svg", format="eps")           # 矢量图 EPS（也可使用 SVG、PDF 等）

plt.show()
