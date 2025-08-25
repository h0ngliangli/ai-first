import torch
import torchvision
import matplotlib.pyplot as plt
import PIL
import numpy as np
import tqdm
import os.path


class SimpleCNN(torch.nn.Module):
    def __init__(self, model_path="model-21.4.pth"):
        super().__init__()
        # output size = (input size - kernel size + 2 x padding) / stride + 1
        # 对于kernel_size=3, padding=1, stride=1的情况, output size = input size
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc = torch.nn.Linear(32 * 7 * 7, 10)
        self.model_path = model_path

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))  # (batch_size, 8, 28, 28)
        x2 = torch.relu(self.conv2(x1))  # (batch_size, 16, 28, 28)
        x3 = self.pool(x2)  # (batch_size, 16, 14, 14)
        x4 = torch.relu(self.conv3(x3))  # (batch_size, 32, 14, 14)
        x5 = self.pool(x4)  # (batch_size, 32, 7, 7)
        x6 = x5.view(x5.size(0), -1)  # (batch_size, 32 * 7 * 7)
        x7 = self.fc(x6)  # (batch_size, 10)
        # print(f'Input shape: {x.shape}')
        # print(f'x1.shape = {x1.shape}')
        # print(f'x2.shape = {x2.shape}')
        # print(f'x3.shape = {x3.shape}')
        # print(f'x4.shape = {x4.shape}')
        # print(f'x5.shape = {x5.shape}')
        # print(f'x6.shape = {x6.shape}')
        return x7, [x1, x2, x3, x4, x5]

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        self.load_state_dict(torch.load(self.model_path))


def prepare_data():
    # 数据预处理
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # 加载数据
    trainset = torchvision.datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    # save first test-images to data/MNIST/test-images
    os.makedirs("data/MNIST/test-images", exist_ok=True)
    for i in range(5):
        img = trainset[i][0]
        img = torchvision.transforms.ToPILImage()(img)
        img.save(f"data/MNIST/test-images/{i}-{trainset[i][1]}.png")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    return trainloader

def main():
    trainloader = prepare_data()
    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    if os.path.exists(model.model_path):
        y = input(f"load model from {model.model_path} (Y/n)?")
        if y.lower() == "y" or y == "":
            model.load()
    else:
        print("开始训练...")
        for epoch in range(3):
            model.train()
            for images, labels in tqdm.tqdm(trainloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        model.save()

    # 读取测试图片
    img_path = "data/MNIST/test-images/0-5.png"
    img = PIL.Image.open(img_path).convert("L").resize((28, 28))
    img_tensor = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(device)
    img_tensor = torchvision.transforms.Normalize((0.5,), (0.5,))(img_tensor)
    # 设置字体为 SimHei（黑体）
    # plt.rcParams["font.sans-serif"] = ["SimHei"]
    # 避免负号显示问题
    # plt.rcParams["axes.unicode_minus"] = False

    # 前向传播并获取各层特征图
    model.eval()
    with torch.no_grad():
        outputs, features = model(img_tensor)

    print(f"Model output: {outputs}")
    # 展示每个卷积层的特征图
    # plt.figure(figsize=(12, 2))
    layer_names = ["image", "conv1", "conv2", "conv2 pooled", "conv3", "conv3 pooled"]
    num_layers = len(layer_names)
    plt.figure(figsize=(20, num_layers * 4))
    # display the original image
    plt.subplot(num_layers, 1, 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Original Image")

    for i, feat in enumerate(features):
        feat = feat.squeeze(0).cpu()  # 去掉batch维度，(num_channels, height, width)
        # print(f"Layer {layer_names[i]}: {feat.shape}")
        num_channels = feat.shape[0]
        for j in range(num_channels):
            plt.subplot(num_layers, num_channels, (i + 1) * num_channels + j + 1)
            plt.imshow(feat[j], cmap="gray")
            plt.axis("off")
            # plt.title(f"Channel {j}")
    plt.suptitle(f"cnn on {img_path}", fontsize=16)
    # maximize the window
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()  # 调整子图间距
    plt.show()

if __name__ == "__main__":
    main()