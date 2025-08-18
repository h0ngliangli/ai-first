# 读取模型文件，用png图像进行测试

import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# 定义图像预处理转换
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图像大小为28x28
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 标准化
])

# 加载训练好的模型
model = torch.load('fashion_mnist_nn.pth')
print(type(model))
# model.eval()  # 设置模型为评估模式

# 定义标签映射
label_to_str = {
    0: "Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# 测试图像目录
test_images_dir = 'data/FashionMNIST/raw/test_images'
# 遍历测试图像目录
for filename in os.listdir(test_images_dir):
    if filename.endswith('.png'):
        # 加载图像
        img_path = os.path.join(test_images_dir, filename)
        image = Image.open(img_path).convert('L')  # 转换为灰度图像
        image = transform(image).unsqueeze(0)  # 添加批次维度

        # 进行预测
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            label = predicted.item()

        # 打印预测结果
        print(f'Image: {filename}, Predicted label: {label_to_str[label]}')