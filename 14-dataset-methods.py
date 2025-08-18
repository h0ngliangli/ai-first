import torch
import torchvision

# FashionMNIST返回
ds = torchvision.datasets.FashionMNIST(
    root='data', download=True, train=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ])
)

# 探索 training_dataset 的方法和属性
print("=== training_dataset 的主要方法和属性 ===")
print(f"类型: {type(ds)}")
print(f"数据集大小: {len(ds)}")
print(f"类别数量: {len(ds.classes)}")
print(f"类别名称: {ds.classes}")
print(f"类别到索引映射: {ds.class_to_idx}")

# 获取第一个样本
image, label = ds[0]
print(f"\n第一个样本:")
print(f"图像形状: {image.shape}") # Size
print(f"图像类型: {type(image)}") # Tensor
print(f"标签: {label} ({ds.classes[label]})")

# 探索所有方法
print(f"\n=== 所有可用的方法和属性 ===")
methods = [method for method in dir(ds) if not method.startswith('_')]
for method in sorted(methods):
    print(f"- {method}")

print(f"\n=== 主要属性的值 ===")
print(f"root: {ds.root}")
print(f"train: {ds.train}")
print(f"transform: {ds.transform}")
print(f"target_transform: {ds.target_transform}")
