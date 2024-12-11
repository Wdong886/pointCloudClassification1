import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 加载数据
data = np.loadtxt("sampleData/sample.txt")
X = data[:, [6, 7, 9, 10, 11, 12,  14, 15]]
y = data[:, 16]

# 检查数据中的缺失值并进行填充
print("X中缺失值的数量:", np.isnan(X).sum())
print("y中缺失值的数量:", np.isnan(y).sum())

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化训练集和测试集
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用StratifiedKFold进行交叉验证
clf = RandomForestClassifier(n_estimators=100, random_state=42)
skf = StratifiedKFold(n_splits=5)

# 计算AUC得分
cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='roc_auc_ovr')  # 'roc_auc_ovr' 支持多分类AUC
print(f"交叉验证AUC得分: {cv_scores}")
print(f"AUC平均值: {cv_scores.mean()}")

# 查看特征重要性
clf.fit(X_train, y_train)
feature_importances = clf.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

print("特征重要性:")
for i in sorted_indices:
    print(f"特征 {i} 的重要性: {feature_importances[i]:.4f}")