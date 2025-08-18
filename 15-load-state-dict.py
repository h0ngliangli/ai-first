import torch
import torchvision

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(28 * 28, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)
model.load_state_dict(torch.load('temp-state-dict.dat'))
# 读取model的参数
test_dataset = torchvision.datasets.FashionMNIST(
    root='data', download=False, train=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
)

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64)

count = 0
total = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        outputs = model(images)
        _, index = torch.max(outputs, dim=1)
        count += torch.sum(index == labels).item()
        total += labels.size(0)

print(f'{count} / {total}')