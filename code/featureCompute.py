import numpy as np
import open3d as o3d
from scipy.stats import skew
from sklearn.neighbors import KDTree
import colorsys

# 读取PCD文件
pcd = o3d.io.read_point_cloud("building.pcd")
points = np.asarray(pcd.points)  # (N, 3)
colors = np.asarray(pcd.colors)  # (N, 3)，假设颜色归一化到[0,1]

# 转换颜色为0-255整数
colors = (colors * 255).astype(int)

# 初始化特征列表
features = []

# 构建KD Tree
tree = KDTree(points)

# 邻域半径（单位与点云一致，例如5cm则0.05）
radius = 0.05

# 用户自定义标签（假设每个点都使用同一标签，如1）
label = 0

for i, (pt, color) in enumerate(zip(points, colors)):
    x, y, z = pt
    R, G, B = color

    # RGB转HSV（返回的H在0-1范围内，可乘以360转换为角度）
    h, s, v = colorsys.rgb_to_hsv(R / 255.0, G / 255.0, B / 255.0)
    H = h * 360
    S = s * 100
    V = v * 100

    # 计算绿度Gr，注意分母加小值防止除零
    Gr = G / (R + G + B + 1e-6)

    # 邻域搜索
    idx = tree.query_radius(pt.reshape(1, -1), r=radius)[0]
    neigh_points = points[idx]

    if len(neigh_points) < 3:
        linearity = planarity = scatter = 0.0
        mean_z = std_z = skew_z = 0.0
    else:
        # 计算邻域的协方差矩阵
        centroid = np.mean(neigh_points, axis=0)
        cov_matrix = np.cov(neigh_points.T)
        # 求特征值并降序排序
        eig_vals, _ = np.linalg.eig(cov_matrix)
        eig_vals = np.sort(eig_vals)[::-1]
        eig_sum = np.sum(eig_vals) + 1e-6
        lam1, lam2, lam3 = eig_vals / eig_sum

        # 计算形态特征
        linearity = (lam1 - lam2) / lam1 if lam1 != 0 else 0.0
        planarity = (lam2 - lam3) / lam1 if lam1 != 0 else 0.0
        scatter = lam3 / lam1 if lam1 != 0 else 0.0

        # 高程信息
        z_vals = neigh_points[:, 2]
        mean_z = np.mean(z_vals)
        std_z = np.std(z_vals)
        skew_z = skew(z_vals)

    # 组合特征
    # 保留3位小数的浮点型特征：X, Y, Z, H, S, V, Gr, Linearity, Planarity, Scatter, mean_z, std_z, skew_z
    # 整数特征：R, G, B, label
    point_features = {
        "X": x, "Y": y, "Z": z,
        "R": R, "G": G, "B": B,
        "H": H, "S": S, "V": V,
        "Gr": Gr,
        "Linearity": linearity,
        "Planarity": planarity,
        "Scatter": scatter,
        "mean_z": mean_z,
        "std_z": std_z,
        "skew_z": skew_z,
        "label": label
    }
    features.append(point_features)

# 定义输出文件格式，每行顺序为：
# X Y Z R G B H S V Gr Linearity Planarity Scatter mean_z std_z skew_z label
with open("building.txt", "w") as f:
    for feat in features:
        # 浮点数按3位小数格式化，整数直接转换
        line = "{:.3f} {:.3f} {:.3f} {} {} {} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {}".format(
            feat["X"], feat["Y"], feat["Z"],
            feat["R"], feat["G"], feat["B"],
            feat["H"], feat["S"], feat["V"],
            feat["Gr"],
            feat["Linearity"], feat["Planarity"], feat["Scatter"],
            feat["mean_z"], feat["std_z"], feat["skew_z"],
            feat["label"]
        )
        f.write(line + "\n")
