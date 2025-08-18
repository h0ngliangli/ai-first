# 用神经网络测试data/FashionMNIST/test-images/目录下的图片
import torch
import torchvision
from fashion_nn import FashionMNISTNN
import os

model = FashionMNISTNN()
model.load_state_dict(torch.load('fashion_mnist_nn.pth'))

img_dir = 'data/FashionMNIST/test-images/'
with torch.no_grad():
    for file_name in sorted(os.listdir(img_dir)):
        if file_name.endswith('.png'):
            image = torchvision.io.decode_image(os.path.join(img_dir, file_name))
            image = image.unsqueeze(0).float() / 255.0
            image = torchvision.transforms.Normalize((0.5,),(0.5,))(image)
            output = model(image)
            output = output.view(-1)
            _, predicted_idx = torch.max(input=output, dim=0)
            predicted_label = torchvision.datasets.FashionMNIST.classes[predicted_idx.item()]
            print(f'{file_name}: max idx: {predicted_idx} {predicted_label}')

