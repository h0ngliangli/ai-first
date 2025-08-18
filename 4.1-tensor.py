import torch

# torch.nn.MSELoss() 返回loss函数
loss_fn = torch.nn.MSELoss(reduction='sum')
y = torch.tensor([1, 2, 3])
y_pred = torch.tensor([1.5, 2.5, 3.5])
# loss函数接收两个参数：预测值和真实值，
# 返回预测值和真实值之间的均方误差
loss = loss_fn(y_pred, y)
print(f'Loss: {loss.item()}')