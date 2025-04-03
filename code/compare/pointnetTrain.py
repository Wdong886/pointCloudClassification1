import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet.model import PointNetDenseCls, feature_transform_regularizer

# ---------------------------
# 自定义数据集类
# ---------------------------
def convert_token(s):
    """
    将输入 token 转换为浮点数：
      - 如果 token 包含 'j'，则转换为复数并取其实部；
      - 否则直接转换为 float。
    注意：s 可能是 bytes 类型，需要先 decode 成字符串。
    """
    token = s.decode('utf-8') if isinstance(s, bytes) else str(s)
    if 'j' in token.lower():
        try:
            return complex(token).real
        except Exception as e:
            raise ValueError(f"转换复数失败, token: {token}, 错误: {e}")
    try:
        return float(token)
    except Exception as e:
        raise ValueError(f"转换浮点数失败, token: {token}, 错误: {e}")

class ScenePointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=2048, transform=None):
        """
        root_dir: 存放txt文件的目录，每个txt文件为一个样本，格式为每行 "X Y Z label" 或更多列数据，其中最后一列为标签
        num_points: 采样点数（不足时可重复采样，多余时随机采样）
        transform: 数据增强（例如旋转、平移）等预处理函数
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.transform = transform
        self.files = glob.glob(os.path.join(root_dir, "*.txt"))
        if len(self.files) == 0:
            raise RuntimeError("未找到任何txt文件，请检查数据目录！")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # 先读取第一行，确定列数
        with open(file_path, "r") as f:
            first_line = f.readline().strip()
            num_cols = len(first_line.split())
        # 构造 converters 字典，对每一列都使用 convert_token
        converters = {i: convert_token for i in range(num_cols)}
        # 使用 np.genfromtxt 加载数据，并自动转换每个 token
        data = np.genfromtxt(file_path, converters=converters)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        # 假设前 3 列为坐标，最后一列为标签（其余列忽略），可以根据实际情况调整
        points = data[:, :3]  # (N, 3)
        labels = data[:, -1].astype(np.int64)  # (N,)

        # 如果点数不足，则重复采样；如果超过，则随机采样
        if points.shape[0] < self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points - points.shape[0], replace=True)
            extra_points = points[choice, :]
            extra_labels = labels[choice]
            points = np.concatenate([points, extra_points], axis=0)
            labels = np.concatenate([labels, extra_labels], axis=0)
        else:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice, :]
            labels = labels[choice]

        # 归一化：使点云均值为 0，点均归一到单位球内
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / m

        if self.transform:
            points = self.transform(points)

        # 转换成 tensor，注意输入要求 shape = (3, num_points)
        points = torch.from_numpy(points.astype(np.float32)).transpose(0, 1)  # (3, num_points)
        labels = torch.from_numpy(labels.astype(np.int64))  # (num_points)
        return points, labels

# ---------------------------
# 训练与验证函数
# ---------------------------
def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    for points, target in train_loader:
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()
        # 对于分割任务，模型输出 (B, num_points, num_classes)
        output, trans, trans_feat = model(points)
        # 将输出 reshape 成 (B*num_points, num_classes)
        output = output.view(-1, output.shape[-1])
        target = target.view(-1)
        loss = criterion(output, target)
        # 如果使用了特征变换正则化，也加上正则项
        if trans_feat is not None:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * points.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for points, target in test_loader:
            points, target = points.to(device), target.to(device)
            output, trans, trans_feat = model(points)
            output = output.view(-1, output.shape[-1])
            target = target.view(-1)
            loss = criterion(output, target)
            if trans_feat is not None:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            running_loss += loss.item() * points.size(0)
            pred = output.max(1)[1]
            correct += pred.eq(target).sum().item()
            total += target.numel()
    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

# ---------------------------
# 主函数
# ---------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PointNet Scene Point Cloud Segmentation")
    parser.add_argument("--data_root", type=str, default=r"data/sample/fixed", help="存放txt点云文件的目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_points", type=int, default=2048, help="每个样本采样的点数")
    parser.add_argument("--epochs", type=int, default=300, help="训练的总轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--num_classes", type=int, default=6, help="类别数，根据数据调整")
    parser.add_argument("--train_split", type=float, default=0.7, help="训练集比例")
    args = parser.parse_args()

    # 检查是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 加载所有数据
    dataset = ScenePointCloudDataset(root_dir=args.data_root, num_points=args.num_points)
    # 划分训练与测试
    num_train = int(len(dataset) * args.train_split)
    num_test = len(dataset) - num_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 加载模型，注意此处用的是分割网络，输出类别数设置为 args.num_classes
    model = PointNetDenseCls(k=args.num_classes, feature_transform=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        test_loss, test_acc = evaluate(model, criterion, test_loader, device)
        print("Epoch [%d/%d]: Train Loss: %.4f, Test Loss: %.4f, Test Acc: %.4f"
              % (epoch+1, args.epochs, train_loss, test_loss, test_acc))
        # 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "pointnet_model.pth")
            print("保存最佳模型，Acc: %.4f" % best_acc)

if __name__ == "__main__":
    main()
