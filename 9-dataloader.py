import torchvision
import torch

training_data = torchvision.datasets.FashionMNIST(
    root='data', # 本地存储路径
    train=True,
    download=True, 
    transform=torchvision.transforms.ToTensor()
)

test_data = torchvision.datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=training_data,
    batch_size=64,)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=64,)

for X, y in train_dataloader:
    print(f'X shape: {X.shape}, y shape: {y.shape}')
    break
print(f'Number of training samples: {len(training_data)}')
print(f'Number of test samples: {len(test_data)}')
print(f'Number of batches in training dataloader: {len(train_dataloader)}')
print(f'Number of batches in test dataloader: {len(test_dataloader)}')
print(f'Batch size: {train_dataloader.batch_size}')