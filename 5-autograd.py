# torch 的 autograd 可以自动计算梯度
# if x.requires_grad=True, then x.grad 就是 x 的梯度

import torch
import math

dtype = torch.float
device = torch.device('cpu')

x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype, device=device)
y = torch.sin(x)

a = torch.randn((), dtype=dtype, device=device, requires_grad=True)
b = torch.randn((), dtype=dtype, device=device, requires_grad=True)
c = torch.randn((), dtype=dtype, device=device, requires_grad=True)
d = torch.randn((), dtype=dtype, device=device, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b * x + c * x**2 + d * x**3
    loss = torch.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss.item(), a.item(), b.item(), c.item(), d.item())

    loss.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        # 清零梯度
        a.grad.zero_()
        b.grad.zero_()
        c.grad.zero_()
        d.grad.zero_()

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')