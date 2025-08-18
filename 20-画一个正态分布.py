import torch
import matplotlib.pyplot as plt

data = torch.randn(100000)

plt.hist(data.numpy(), bins=100, density=True)
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()