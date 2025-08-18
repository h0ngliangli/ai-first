import torch

model = torch.nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    inputs = torch.randn(10, 3)  # 10 samples, 3 features
    targets = torch.randn(10, 1)  # 10 target values

    optimizer.zero_grad()  # Clear gradients
    outputs = model(inputs)  # Forward pass
    loss = torch.nn.functional.mse_loss(outputs, targets)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update parameters

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
print(f'Final model parameters: {model.weight.data}, {model.bias.data}')