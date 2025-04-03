import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from Pointnet2.models.pointnet2_sem_seg_msg import get_model
from torch import nn


def convert_complex_to_float(s):
    try:
        s = s.decode('utf-8') if isinstance(s, bytes) else s
        return np.real(complex(s))
    except Exception as e:
        print(f"转换错误: {s}, 错误信息: {e}")
        return 0.0


def compute_normals(points):
    normals = np.random.rand(points.shape[0], 3) * 2 - 1
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def compute_colors(points):
    return np.ones((points.shape[0], 3))

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

class ScenePointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=2048, debug=False):
        self.num_points = num_points
        self.debug = debug
        self.point_clouds = []  # 存储点云特征 (9, num_points)
        self.point_labels = []  # 存储逐点标签 (num_points,)

        # 读取所有点云文件
        files = glob.glob(os.path.join(root_dir, "*.txt"))
        if not files:
            raise ValueError("未找到任何点云文件，请检查目录路径！")

        for file in files:
            data = np.genfromtxt(file, converters={10: lambda s: np.real(complex(s.decode()))})
            points = data[:, :3]  # 提取XYZ坐标
            labels = data[:, -1].astype(np.int64)  # 最后一列为标签

            # 下采样或填充至固定点数
            if points.shape[0] >= self.num_points:
                indices = np.random.choice(points.shape[0], self.num_points, replace=False)
                points = points[indices]
                labels = labels[indices]
            else:
                pad_size = self.num_points - points.shape[0]
                points = np.vstack([points, np.zeros((pad_size, 3))])  # 填充坐标
                labels = np.concatenate([labels, -np.ones(pad_size, dtype=np.int64)])  # 填充标签为-1

            # 计算法线和颜色（示例代码，需根据实际数据调整）
            normals = np.random.randn(points.shape[0], 3)  # 随机法线
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # 单位化
            colors = np.ones((points.shape[0], 3))  # 假设颜色全为白色

            # 组合特征 (9维: XYZ + 法线 + 颜色)
            features = np.hstack([points, normals, colors]).T  # (9, num_points)
            self.point_clouds.append(features.astype(np.float32))
            self.point_labels.append(labels.astype(np.int64))

            # 验证数据维度
            assert features.shape == (9, self.num_points), f"特征维度错误: {features.shape}"
            assert labels.shape == (self.num_points,), f"标签维度错误: {labels.shape}"

        if self.debug:
            print(f"成功加载 {len(self.point_clouds)} 个点云，每个点云包含 {self.num_points} 个点")

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.point_clouds[idx]),  # (9, num_points)
            torch.from_numpy(self.point_labels[idx])    # (num_points,)
        )


def split_dataset(dataset, train_ratio=0.8):
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    split = int(num_samples * train_ratio)
    return Subset(dataset, indices[:split]), Subset(dataset, indices[split:])


def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (points, labels) in enumerate(train_loader):
        points = points.to(device)  # (B, 9, N)
        labels = labels.to(device)  # (B, N)

        optimizer.zero_grad()
        pred, _ = model(points)  # 输出形状: (B, N, num_classes)

        # 调整维度并计算损失
        loss = criterion(pred.permute(0, 2, 1), labels)  # CrossEntropyLoss需要 (B, C, N)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 打印第一个batch的预测结果
        if batch_idx == 0:
            pred_labels = pred.argmax(dim=-1)
            print(f"训练Batch 0 - 预测标签: {pred_labels[0, :5].cpu().numpy()}")
            print(f"训练Batch 0 - 真实标签: {labels[0, :5].cpu().numpy()}")

    return total_loss / len(train_loader)


def evaluate(model, criterion, test_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in test_loader:
            points = points.to(device)
            labels = labels.to(device)

            pred, _ = model(points)  # (B, N, num_classes)
            loss = criterion(pred.permute(0, 2, 1), labels)
            total_loss += loss.item()

            # 计算准确率（忽略填充点）
            pred_labels = pred.argmax(dim=-1)
            valid_mask = labels != -1
            correct += (pred_labels[valid_mask] == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return total_loss / len(test_loader), accuracy


def main():
    data_root = "data/sample/txt"
    batch_size = 4
    num_epochs = 50
    num_classes = 5  # 根据实际类别数修改
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化数据集和数据加载器
    dataset = ScenePointCloudDataset(data_root, num_points=2048, debug=True)
    train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型、优化器和损失函数
    model = get_model(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略填充标签

    # 训练循环
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        test_loss, test_acc = evaluate(model, criterion, test_loader, device)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | 训练损失: {train_loss:.4f} | 测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.4f}")


if __name__ == "__main__":
    main()