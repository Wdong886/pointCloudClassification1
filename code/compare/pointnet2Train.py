import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Pointnet2.models.pointnet2_sem_seg_msg import get_model, get_loss  # 你的PointNet++模型

def compute_normals(points):
    """
    计算法向量（这里使用随机法向量，如果有真实法向量可替换）
    """
    normals = np.random.rand(points.shape[0], 3) * 2 - 1  # 生成 -1 到 1 之间的随机值
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # 归一化
    return normals

def compute_colors(points):
    """
    计算颜色信息（这里假设全部是白色，如果有真实 RGB 可替换）
    """
    colors = np.ones((points.shape[0], 3))  # 统一为白色 (1,1,1)
    return colors


class ScenePointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=2048):
        self.root_dir = root_dir
        self.num_points = num_points
        self.files = glob.glob(os.path.join(root_dir, "*.txt"))
        if len(self.files) == 0:
            raise RuntimeError("未找到任何 txt 文件，请检查数据目录！")

    def __len__(self):  # 添加这个方法
        return len(self.files)

    def __getitem__(self, idx):
        data = np.loadtxt(self.files[idx])  # 读取点云
        points = data[:, :3]  # XYZ坐标
        labels = data[:, -1].astype(np.int64)  # 标签

        # 增加额外特征（如法向量、颜色）
        normals = compute_normals(points)  # 计算法向量
        colors = compute_colors(points)  # 计算颜色

        # 组合成 12 维输入
        features = np.concatenate([points, normals, colors], axis=1)  # (N, 12)

        # 采样点云
        if features.shape[0] < self.num_points:
            choice = np.random.choice(features.shape[0], self.num_points - features.shape[0], replace=True)
            extra_features = features[choice, :]
            extra_labels = labels[choice]
            features = np.concatenate([features, extra_features], axis=0)
            labels = np.concatenate([labels, extra_labels], axis=0)
        else:
            choice = np.random.choice(features.shape[0], self.num_points, replace=False)
            features = features[choice, :]
            labels = labels[choice]

        # 归一化点云
        centroid = np.mean(features[:, :3], axis=0)
        features[:, :3] = features[:, :3] - centroid
        max_dist = np.max(np.sqrt(np.sum(features[:, :3] ** 2, axis=1)))
        features[:, :3] = features[:, :3] / max_dist

        # 转换成 PyTorch Tensor
        features = torch.from_numpy(features.astype(np.float32)).transpose(0, 1)  # (12, num_points)
        labels = torch.from_numpy(labels.astype(np.int64))  # (num_points)

        return features, labels


def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    for points, labels in train_loader:
        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad()
        pred, _ = model(points)  # (B, N, num_classes)

        # 修正 reshape 方式
        pred = pred.contiguous().reshape(-1, pred.shape[-1])  # (B*N, num_classes)
        labels = labels.view(-1)  # (B*N,)

        loss = criterion(pred, labels, None, None)  # 计算损失
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)



def evaluate(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)
            pred, _ = model(points)  # (B, N, num_classes)

            # 修正 reshape 方式
            pred = pred.contiguous().reshape(-1, pred.shape[-1])  # (B*N, num_classes)
            labels = labels.view(-1)  # (B*N,)

            loss = criterion(pred, labels, None, None)
            running_loss += loss.item()
            pred_labels = pred.argmax(dim=1)
            correct += (pred_labels == labels).sum().item()
            total += labels.numel()
    accuracy = correct / total
    return running_loss / len(test_loader), accuracy


# ---------------------------
# 训练主函数
# ---------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PointNet++ Scene Point Cloud Segmentation")
    parser.add_argument("--data_root", type=str, default=r"C:\Users\Lenovo\Desktop\待分类点云\1", help="点云 txt 文件目录")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_points", type=int, default=2048, help="采样点数")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--num_classes", type=int, default=6, help="类别数")
    args = parser.parse_args()

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    dataset = ScenePointCloudDataset(root_dir=args.data_root, num_points=args.num_points)
    num_train = int(len(dataset) * 0.6)
    num_test = len(dataset) - num_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 加载模型
    model = get_model(args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = get_loss()

    # 训练
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        test_loss, test_acc = evaluate(model, criterion, test_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "pointnet2_model_new.pth")
            print(f"保存最佳模型，Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
