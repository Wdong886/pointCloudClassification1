import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据
# 假设数据为 .txt 格式，使用空格分隔
# 每行的前三个数字是XYZ坐标，不作为特征
# 1. 打开原文件，读取内容
with open('sampleData/sample.txt', 'r') as f:
    lines = f.readlines()

# 2. 去除每行结尾的空格
cleaned_lines = [line.rstrip() for line in lines]

# 3. 将处理后的内容写入新的文件
with open('sampleData/sample_clean.txt', 'w') as f:
    f.writelines("\n".join(cleaned_lines))

# 4. 读取处理后的数据
data = pd.read_csv('sampleData/sample_clean.txt', delimiter=' ', header=None)

# 提取特征和标签
X = data.iloc[:, 3:16]  # 第4到16列是特征
y = data.iloc[:, 16]    # 第17列是标签

# 2. 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. sCARS特征选择（简化为Lasso回归）
def sCARS(X_train, y_train, n_features=10):
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train, y_train)
    selected_features = np.where(lasso.coef_ != 0)[0]  # 选择非零系数的特征
    return selected_features

# 4. SPA特征选择（简化为PCA投影）
def SPA(X_train, selected_features, n_components=5):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    X_selected = X_train.iloc[:, selected_features]
    X_pca = pca.fit_transform(X_selected)
    return X_pca, pca.components_

# 5. ICO特征优化（简化为方差选择）
def ICO(X_train, selected_features, n_iter=10):
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.1)  # 去掉方差小于0.1的特征
    X_selected = X_train.iloc[:, selected_features]
    X_optimized = selector.fit_transform(X_selected)
    return X_optimized

# 6. 特征筛选流程
# 第一阶段：sCARS特征选择
selected_features_scars = sCARS(X_train, y_train)
print(f"sCARS 选择的特征索引: {selected_features_scars}")
print(f"sCARS 选择的特征数据:\n{X_train.iloc[:, selected_features_scars].head()}")

# 第二阶段：SPA投影处理
X_train_spa, pca_components = SPA(X_train, selected_features_scars)
print(f"SPA 选择的PCA成分: \n{pca_components}")
print(f"SPA 降维后的特征数据:\n{X_train_spa[:5]}")  # 显示前五行

# 第三阶段：ICO优化
X_train_ico = ICO(X_train, selected_features_scars)
print(f"ICO 优化后的特征数据:\n{X_train_ico[:5]}")  # 显示前五行

# 可以将特征输出保存到文件中，如果需要：
# np.savetxt("selected_features_scars.txt", X_train.iloc[:, selected_features_scars].values)
# np.savetxt("X_train_spa.txt", X_train_spa)
# np.savetxt("X_train_ico.txt", X_train_ico)

