import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1,2,3])
# xx = [x, x**2, x**3]
# unsqueeze(-1)将x的最后一维扩展为1维
# pow(p)将x的每个元素分别取p的幂
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    # 3维输入，1维输出，对应 
    # y = a + b * x + c * x^2 + d * x^3
    # a是偏置bias，b, c, d 是权重weight
    torch.nn.Linear(3,1),
    # 
    torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):
    # 
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)
    if t % 500 == 499:
        print(t, loss.item(), model[0].weight, model[0].bias)
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            param.grad.zero_()

linear_layer = model[0]
a = linear_layer.bias.item()
b, c, d = linear_layer.weight[0].tolist()
print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')