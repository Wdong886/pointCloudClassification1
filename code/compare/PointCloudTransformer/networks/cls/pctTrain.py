import os
import glob
import numpy as np
import jittor as jt
from jittor import nn, init
from jittor.dataset import Dataset, DataLoader
import argparse

# 从 pct.py 导入模型，这里假设文件 pct.py 与本脚本在同一目录下
from pct import Point_Transformer


# ---------------------------
# 自定义数据集类
# ---------------------------
class ScenePointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, transform=None):
        """
        root_dir: 存放 txt 文件的目录，每个文件为一个样本，格式：每行 "X Y Z label"
        num_points: 每个点云采样的点数
        transform: 可选的数据增强
        """
        super().__init__()
        self.root_dir = root_dir
        self.num_points = num_points
        self.transform = transform
        self.files = glob.glob(os.path.join(root_dir, "*.txt"))
        if len(self.files) == 0:
            raise RuntimeError("未找到任何 txt 文件，请检查数据目录！")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.loadtxt(self.files[index])
        if data.shape[1] != 4:
            raise ValueError(f"{self.files[index]} 格式错误，应该为 'X Y Z label'")
        # 提取点坐标和标签
        points = data[:, :3]  # (N, 3)
        labels = data[:, 3].astype(np.int64)  # (N,)
        # 假设全局标签一致，取第一个作为样本的标签
        global_label = labels[0]

        # 采样固定数量点
        if points.shape[0] >= self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[choice, :]

        # 归一化：平移到中心并归一化到单位球内
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / max_dist

        # 如果需要数据增强，可在此处加入 transform 处理
        if self.transform:
            points = self.transform(points)

        # 转换为 jittor 数组，模型要求输入形状 (3, num_points)
        points = points.astype(np.float32).T  # (3, num_points)
        return jt.array(points), global_label


# ---------------------------
# 训练和验证函数
# ---------------------------
def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for points, label in train_loader:
        # points: (B, 3, num_points), label: (B,)
        logits = model(points)  # 输出形状 (B, num_classes)
        loss = criterion(logits, jt.array(label))
        optimizer.step(loss)
        total_loss += loss.item()
        pred = jt.argmax(logits, dim=1)
        total_correct += (pred == jt.array(label)).sum().item()
        total_samples += len(label)
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def eval_epoch(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with jt.no_grad():
        for points, label in val_loader:
            logits = model(points)
            loss = criterion(logits, jt.array(label))
            total_loss += loss.item()
            pred = jt.argmax(logits, dim=1)
            total_correct += (pred == jt.array(label)).sum().item()
            total_samples += len(label)
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ---------------------------
# 主函数
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Point_Transformer 点云分类训练")
    parser.add_argument("--data_root", type=str, default="data/testdata", help="存放 txt 点云文件的目录")
    parser.add_argument("--num_points", type=int, default=1024, help="采样点数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--num_classes", type=int, default=5, help="类别数")
    args = parser.parse_args()

    jt.flags.use_cuda = 0
    print("使用 jittor，是否启用 CUDA：", jt.flags.use_cuda)

    # 构造数据集并划分训练/测试集（这里80%用于训练，20%用于测试）
    dataset = ScenePointCloudDataset(root_dir=args.data_root, num_points=args.num_points)
    num_train = int(len(dataset) * 0.8)
    num_test = len(dataset) - num_train
    train_dataset, test_dataset = dataset.split([num_train, num_test])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 创建模型，输出通道数即类别数
    model = Point_Transformer(output_channels=args.num_classes)

    # 定义优化器和损失函数
    optimizer = nn.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_epoch(model, test_loader, criterion)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            model.save("best_model.pkl")
            print("保存最佳模型，准确率：", best_acc)


if __name__ == '__main__':
    main()
