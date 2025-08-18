# 导出FashionMNIST前十个测试图片

import torch
import torchvision

classes = torchvision.datasets.FashionMNIST.classes.copy()
# replace / to _
classes = [c.replace('/', '_') for c in classes]

test_dataset = torchvision.datasets.FashionMNIST(
    root='data', train=False, download=False,
)

for i in range(10):
    image, label_idx = test_dataset[i]
    label = classes[label_idx]
    image.save(f'data/FashionMNIST/test-images/{i}-{label}.png')
    