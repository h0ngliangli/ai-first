import torch
import torchvision

model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=1),  # (batch, 1, 28, 28) => (batch, 1 * 28 * 28)
    torch.nn.Linear(28 * 28, 128),  # (batch, 28 * 28) => (batch, 128)
    torch.nn.ReLU(),  # (batch, 128) => (batch, 128)
    torch.nn.Linear(128, 64),  # (batch, 128) => (batch, 64)
    torch.nn.ReLU(),  # (batch, 64) => (batch, 64)
    torch.nn.Linear(64, 10),  # (batch, 64) => (batch, 10)
    # torch.nn.ReLU(),
    # torch.nn.Linear(10, 1)
)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),  # scale to [0, 1]
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = torchvision.datasets.FashionMNIST(
    root="data", download=False, train=True, transform=transform
)

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)

for epoch in range(5):
    print(f"epoch {epoch}")
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)  # (batch)
        # print(f'outputs {outputs}')
        # _, max_indices = torch.max(outputs, dim=1) # (batch)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"loss {loss}")

# 测试模型
test_dataloader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.FashionMNIST(
        root="data", download=False, train=False, transform=transform
    ),
    batch_size=64,
    shuffle=False,
)
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        outputs = model(images) # (batch, 10)
        _, predicted_indices = torch.max(outputs, dim=1) # (batch)
        correct += (predicted_indices == labels).sum().item()
        total += labels.size(0)
print(f'{correct} / {total} {correct / total :.2f}')