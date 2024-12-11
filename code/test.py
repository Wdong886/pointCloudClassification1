import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

# 1. 读取数据
data = pd.read_csv('sampleData/sample.txt', delim_whitespace=True, header=None)  # 替换为你的txt文件路径

# 从第4到第16列作为特征，第17列作为标签
X = data.iloc[:, 6:16]  # 特征数据 (第4到16列)
y = data.iloc[:, 16]  # 标签数据 (第17列)

# 2. 使用LabelEncoder转换标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # 将标签从1-6转换为0-5

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义基模型和堆叠模型
base_models = [
    ('Random Forest', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('SVM', SVC(probability=True, random_state=42)),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=5))
]
meta_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)


# 获取单个模型的学习曲线
def compute_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy', n_jobs=-1, random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    return train_sizes, test_mean


# 获取基模型的学习曲线
plt.figure(figsize=(12, 8))

for name, model in base_models:
    train_sizes, test_mean = compute_learning_curve(model, X_train, y_train)
    plt.plot(train_sizes, test_mean, 'o-', label=f'{name} (Validation Accuracy)')


# 计算堆叠模型的学习曲线
def compute_stacked_learning_curve(base_models, meta_model, X, y, train_sizes=np.linspace(0.1, 1.0, 10)):
    stacked_test_scores = []

    for train_size in train_sizes:
        # 当前训练集大小的索引
        train_sample_size = int(train_size * len(X_train))
        X_sub_train, y_sub_train = X_train[:train_sample_size], y_train[:train_sample_size]

        # 训练基模型并生成预测概率作为新特征
        stacked_features_train = []
        stacked_features_test = []

        for _, base_model in base_models:
            model_clone = clone(base_model)
            model_clone.fit(X_sub_train, y_sub_train)
            stacked_features_train.append(model_clone.predict_proba(X_sub_train))
            stacked_features_test.append(model_clone.predict_proba(X_test))

        stacked_features_train = np.hstack(stacked_features_train)
        stacked_features_test = np.hstack(stacked_features_test)

        # 训练元模型
        meta_model.fit(stacked_features_train, y_sub_train)
        test_score = meta_model.score(stacked_features_test, y_test)
        stacked_test_scores.append(test_score)

    return train_sizes, stacked_test_scores


# 绘制堆叠模型的学习曲线
train_sizes, stacked_test_scores = compute_stacked_learning_curve(base_models, meta_model, X_train, y_train)
plt.plot(train_sizes * len(X_train), stacked_test_scores, 'o-', color='b', label="Stacked Model (Validation Accuracy)")

# 设置图形标签和标题
plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curves of Individual Models and Stacked Model")
plt.legend(loc="best")
plt.grid()

# 保存图像为 JPG 和 EPS 格式
plt.savefig("models_learning_curve.jpg", format="jpg", dpi=300)
plt.savefig("models_learning_curve.eps", format="eps")

# 显示图形
plt.show()
