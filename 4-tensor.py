# torch的tensor 和 numpy 几乎是一样的
# 只需要把 numpy 的函数换成 torch 的函数即可
# 这里的代码是用 tensor 来实现 1-numpy.py 的功能
import torch
import math

dtype = torch.float
device = torch.device('cpu')

x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype, device=device)
y = torch.sin(x)

a = torch.randn((), dtype=dtype, device=device)
b = torch.randn((), dtype=dtype, device=device)
c = torch.randn((), dtype=dtype, device=device)
d = torch.randn((), dtype=dtype, device=device)

learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b * x + c * x**2 + d * x**3
    loss = torch.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss.item(), a.item(), b.item(), c.item(), d.item())
    
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')