import os
import numpy as np
import torch
import torch.nn as nn
import time
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from pointnet.model import PointNetDenseCls  # 确保pointnet库已安装

# ---------------------------
# 配置参数（使用时修改这部分即可）
# ---------------------------
CONFIG = {
    "model_path": "pointnet_model.pth",         # 训练好的模型路径
    "test_file": r"C:\Users\Lenovo\Desktop\待分类点云\1\class_0.txt",  # 待测试的TXT文件路径
    "num_points": 2048,                         # 与训练时一致的采样点数
    "num_classes": 6,                           # 类别数（与训练一致）
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 自动选择设备

    # 滑动窗口参数（在XY平面上采样）
    "window_size": 1.0,  # 窗口大小（单位与点云一致）
    "stride": 0.5,       # 窗口滑动步长
    "min_points": 100    # 窗口内至少要求的点数，否则跳过
}


# ---------------------------
# 点云预处理函数（必须与训练时一致）
# ---------------------------
def preprocess_point_cloud(points, num_points):
    """输入原始点云 (N,3)，输出预处理后的 (3,num_points)"""
    # 如果点数不足，则补采样；如果超过，则随机采样
    if points.shape[0] < num_points:
        choice = np.random.choice(points.shape[0], num_points - points.shape[0], replace=True)
        points = np.concatenate([points, points[choice]], axis=0)
    else:
        choice = np.random.choice(points.shape[0], num_points, replace=False)
        points = points[choice]

    # 归一化（零均值+单位球）
    centroid = np.mean(points, axis=0)
    points = points - centroid
    m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / (m + 1e-10)
    return points.transpose()  # 转换为 (3, num_points)


# ---------------------------
# 加载模型
# ---------------------------
def load_model(model_path, num_classes, device):
    # 必须与训练时参数一致：feature_transform=True
    model = PointNetDenseCls(k=num_classes, feature_transform=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# ---------------------------
# 滑动窗口采样并分类
# ---------------------------
def sliding_window_classification(model, file_path, num_points, device, window_size, stride, min_points):
    # 加载整幅点云数据（假设每行至少包含 "X Y Z ..."）
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    points_all = data[:, :3]  # (N,3)

    # 计算XY平面的边界
    min_xy = np.min(points_all[:, :2], axis=0)
    max_xy = np.max(points_all[:, :2], axis=0)

    results = []  # 用于存储每个窗口的结果

    # 构造滑动窗口
    x_starts = np.arange(min_xy[0], max_xy[0], stride)
    y_starts = np.arange(min_xy[1], max_xy[1], stride)
    window_count = 0

    for x0 in x_starts:
        for y0 in y_starts:
            # 定义窗口范围（这里采用矩形窗口）
            x1 = x0 + window_size
            y1 = y0 + window_size
            # 筛选窗口内的点（只依据XY坐标）
            mask = (points_all[:, 0] >= x0) & (points_all[:, 0] <= x1) & \
                   (points_all[:, 1] >= y0) & (points_all[:, 1] <= y1)
            window_points = points_all[mask, :]  # (n,3)
            if window_points.shape[0] < min_points:
                continue  # 点数太少则跳过

            # 预处理窗口内的点
            processed_points = preprocess_point_cloud(window_points, num_points)
            input_tensor = torch.from_numpy(processed_points.astype(np.float32)).unsqueeze(0).to(device)  # (1,3,num_points)

            # 推理
            with torch.no_grad():
                output, _, _ = model(input_tensor)  # 输出 (1,num_points,num_classes)
            # 对每个采样点获取预测类别，使用多数投票确定整个窗口的类别
            pred_labels = output.argmax(dim=-1).squeeze().cpu().numpy()  # (num_points,)
            majority_label = Counter(pred_labels).most_common(1)[0][0]

            window_count += 1
            results.append({
                "window_id": window_count,
                "x_range": (x0, x1),
                "y_range": (y0, y1),
                "num_points": window_points.shape[0],
                "predicted_label": majority_label
            })

            print(f"窗口 {window_count}: X[{x0:.2f}, {x1:.2f}] Y[{y0:.2f}, {y1:.2f}], 点数: {window_points.shape[0]}, 预测类别: {majority_label}")

    return results


# ---------------------------
# 主程序
# ---------------------------
if __name__ == "__main__":
    print("=" * 50)
    print(f"设备: {CONFIG['device'].upper()}")
    print(f"加载模型: {os.path.basename(CONFIG['model_path'])}")
    model = load_model(CONFIG['model_path'], CONFIG['num_classes'], CONFIG['device'])
    print("开始滑动窗口采样和分类...")
    start_time = time.time()
    results = sliding_window_classification(
        model=model,
        file_path=CONFIG['test_file'],
        num_points=CONFIG['num_points'],
        device=CONFIG['device'],
        window_size=CONFIG['window_size'],
        stride=CONFIG['stride'],
        min_points=CONFIG['min_points']
    )
    total_time = time.time() - start_time
    print("=" * 50)
    print(f"采样和分类完成，共处理窗口数: {len(results)}, 总耗时: {total_time:.2f}s")
