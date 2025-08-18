import torch
import torchvision

# 和17.1相比，增加epoch看对结果是否有影响。
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
for epoch in range(10):
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



# performance
# epoch     conv1     conv2   stride      dropout   time    accuracy    loss1   loss2   loss3   loss4   loss5     loss6   loss7   loss8   loss9   loss10
# 5         16        32      1           0         1:31    90.69%      0.58    0.19    0.27    0.04    0.19
# 10        16        32      1           0         2:49    91.58%      0.52    0.34    0.04    0.34    0.14      0.05    0.12    0.11    0.09    0.08  

# 总结：
# epoch增加 -> loss下降明显但accuracy提升不大，说明模型已经接近拟合极限，
# 因为accuracy有小幅提升，说明模型还在学习，没有过拟合，但边际效应很小。