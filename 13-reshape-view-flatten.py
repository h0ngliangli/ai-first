import torch

x = torch.randn(64, 1, 28, 28)
print(f'x.shape = {x.shape}')

# 方法1：使用view()方法
# view()：PyTorch中调整张量形状的方法，不改变数据内容，仅改变
# 维度。
# -1表示自动计算该维度的大小，
# 28*28表示最后一维展平为784维向量
# 将四维张量 [64, 1, 28, 28] 转换为二维张量
# 最终输出形状：[64, 784]
y = x.view(-1, 28 * 28)
print(f'y.shape = {y.shape}')

# 方法2：使用reshape()方法
z = x.reshape(-1, 28 * 28)
print(f'z.shape = {z.shape}')


# 从第1维开始展平，即保留第0维(64)不变，后面的维度展平
# 这与x.view(-1, 28 * 28)等价
w = x.flatten(start_dim=1)
print(f'w.shape = {w.shape}')

print(torch.equal(y, z))
print(torch.equal(y, w))