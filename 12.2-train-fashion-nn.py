# 使用torch，构建FashionMNIST数据集的神经网络模型
import torch
import torch.nn as nn
import torch.optim as optim
from fashion_nn import FashionMNISTNN  # 导入自定义的神经网络模型

# datasets：提供常用数据集（如FashionMNIST、CIFAR10）的加载接口。
# transforms：包含图像预处理工具（如归一化、裁剪、旋转等）。
from torchvision import datasets, transforms

# 将多个预处理步骤组合成一个顺序执行的管道
transform = transforms.Compose([
    transforms.ToTensor(), # 将PIL图像或numpy.ndarray转换为Tensor，并归一化到[0, 1]范围。
    # 标准化公式 normalized = (x - mean) / std
    # 将[0, 1]映射到[-1, 1]
    # 零中心的数据有助于加快收敛速度
    transforms.Normalize((0.5,), (0.5,)) 
])
# 下载并加载FashionMNIST数据集
# train=True表示加载训练集，train=False表示加载测试集
# download=True表示如果数据集不存在则下载，transform=transform表示应用预处理转换
# len(train_dataset) 返回数据集的大小，每个元素是一个tuple，包含图像和标签
# image, label = train_dataset[0]
# 
train_dataset = datasets.FashionMNIST(root='data', train=True,
                                       download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='data', train=False,
                                      download=True, transform=transform)
# 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64,
                                          shuffle=False)
# 定义神经网络模型

# 实例化模型、定义损失函数和优化器
model = FashionMNISTNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练模型
# 训练次数的组成:
#   Epoch（轮次）：模型完整遍历整个训练集一次称为一个epoch
#   Iteration（迭代）：每次处理一个批量的数据称为一次迭代
# 
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# 测试模型
correct = 0
total = 0
# no_grad() 禁用梯度计算，在此模式下，模型只进行前向推理，不计算梯度
# 这在测试或验证阶段非常有用，可以减少内存消耗和计算
with torch.no_grad():
    # images.shape = [64, 1, 28, 28]
    # labels.shape = [64]
    for images, labels in test_loader:
        # 前向传播
        outputs = model(images)
        # outputs与outputs.data的区别：
        # outputs是一个完整的张量，包含张量的数值和梯度信息
        # outputs.data是一个只包含数值的张量，不包含梯度信息
        # 实际上，在torch.no_grad()上下文中，outputs.data和outputs是等价的
        # 这里只是一种明确的写法。
        # torch.max(tensor, 1) 返回第1维的最大值和对应的索引
        _, max_index = torch.max(outputs.data, 1)
        # size(0) 返回张量的第0维大小，即批量大小
        total += labels.size(0)
        correct += (max_index == labels).sum().item()   
print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
# 保存模型
torch.save(model.state_dict(), 'fashion_mnist_nn.pth')
print('Model saved to fashion_mnist_nn.pth')