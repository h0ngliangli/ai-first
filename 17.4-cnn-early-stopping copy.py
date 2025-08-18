import torch
from torch.utils.data import random_split
import torchvision

# 和17.3相比：
# 加入early stopping
# ✅ 功能说明：
#     自动监控 验证集 loss
#     如果验证 loss 连续若干 epoch 没有改善，就停止训练
#     自动保存 最佳模型参数

class EarlyStopping:
    def __init__(self, patience=3, delta=0.001, path='17.3-best-model.pth')
        self.patience = 


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

# 切分成训练集和验证集（90%训练，10%验证）
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

class CNNFashion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
model = CNNFashion()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(10):
    print(f"Epoch {epoch + 1}")
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # validation
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    print(f"Val   Loss: {avg_val_loss:.4f}, Val   Acc: {val_accuracy:.2f}%")

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
print(f"Test Accuracy: {100 * correct / total:.2f}%")

import matplotlib.pyplot as plt

# Loss 曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

# Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy Curve")
plt.tight_layout()
plt.show()

# 总结：
# 从 epoch 6 之后：
#     验证 loss 上升 + 准确率不再提升 ⟶ 已进入过拟合阶段
#     当前还不严重，但如果你训练到 20 或 50 epoch，可能验证性能会明显下降。