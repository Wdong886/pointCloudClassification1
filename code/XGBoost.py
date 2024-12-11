import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 加载数据
data = np.loadtxt("sampleData/sample.txt")
X = data[:, 6:16]  # 特征数据 (第4到16列, 共13个特征)
y = data[:, 16]    # 标签数据 (第17列)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建并训练 XGBoost 模型
xgb_clf = XGBClassifier(
    n_estimators=5,       # 树的数量
    max_depth=2,            # 每棵树的最大深度
    learning_rate=0.1,      # 学习率
    subsample=0.7,          # 每棵树使用的样本比例
    colsample_bytree=0.7,   # 每棵树使用的特征比例
    random_state=42
)
xgb_clf.fit(X_train, y_train)

# 4. 预测
y_pred = xgb_clf.predict(X_test)

# 5. 计算并输出评价指标
accuracy = accuracy_score(y_test, y_pred)
print(f"模型精度: {accuracy:.4f}")

# 打印详细的分类报告
print("分类报告:")
print(classification_report(y_test, y_pred))

# 打印混淆矩阵
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
