import torch
import torchvision


class CNNFashion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # print(f'{x.shape}')
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 数据预处理和加载
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
dataset = torchvision.datasets.FashionMNIST(
    root="data", train=True, download=False, transform=transform
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
# 初始化模型、损失函数和优化器
model = CNNFashion()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
for epoch in range(5):
    print(f"Epoch {epoch + 1}")
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item():.2f}")


# 测试模型
test_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=False, download=False, transform=transform
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False
)
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {100 * correct / total:.2f}%")


# 总结：
# 1. 通道数增加，准确率未必上升。conv1(1, 16)和conv1(1,32)准确率接近。
# 理论上32通道比16通道参数更多，可以提取更多特征，但FashionMNIST数据集相对简单，
# 说明16通道的卷积层可能已经足够捕捉关键特征，增加通道未必有效.
# 如果任务难度更高（如 CIFAR-10，ImageNet），增大通道数的效果会更明显。
# 2. Loss并没有出现越来越小的趋势，可能是因为训练周期较短，模型尚未收敛。
# Loss的抖动可能来自于
# 数据扰动
# dropout()会随机丢弃部分神经元
# 学习率较高：造成loss更新幅度较大，容易出现震荡。
# 3. Loss震荡但准确率未明显下降.
# 首先，loss和accuracy是不同的衡量，它们的关系并不总是正相关。
# 交叉熵是一个连续损失函数，它关注模型输出的概率分别是否接近1.0的分布。
# Accuracy是一个离散的指标，仅关注最大概率的类别是否匹配目标。
# 
# 但概率可能未接近1.0（比如只有0.6，预测正确，但Loss仍然可能较高）
# 另外一种可能性：模型A预测10个样本，每个正确类别概率为0.6，Loss较高。
# 模型B预测9个样本概率为0.99，1个错误，Loss低，但准确率=90%。
# 建议：
# 延长训练周期：尝试10-20 epochs看是否会收敛

# performance
# conv1     conv2   stride      dropout     time    accuracy    loss1   loss2   loss3   loss4   loss5
# 16        32      1           0           1:59    91.05%      0.38    0.12    0.35    0.11    0.48
# 16        32      1           0.25        1:52    90.54%      0.28    0.28    0.22    0.31    0.25
# 32        64      1           0.25        3:03    91.07%      0.43    0.34    0.21    0.14    0.13
