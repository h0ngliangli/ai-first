# 理解卷积的概念

import torch

input = torch.tensor([[[[1,2], [3,4]], [[5,6], [7,8]]]]).float()  # batch_size=1, channels=2, height=2, width=2

conv_layer = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False)
print(f"卷积层权重:\n{conv_layer.weight}")
print(conv_layer.weight.shape)
with torch.no_grad():
    for i in range(1 * 2 * 2 * 2):
        conv_layer.weight.view(-1)[i] = i + 1  # 手动设置权重以便跟踪（实际中这是学习得到的）
    # conv_layer.weight = torch.tensor([[[[1,1],[1,1]], [[1,1],[1,1]]]])
    print(f"手动设置卷积层权重:\n{conv_layer.weight}")
output = conv_layer(input)

print(output)  # 应该是 [[[[ 204.]]]] = [[[[1*1+2*2+3*3+4*4 + 5*5+6*6+7*7+8*8]]]]

print(1 + 4 + 9 + 16 + 25 + 36 + 49 + 64)