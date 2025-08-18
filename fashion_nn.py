import torch
import torch.nn as nn

class FashionMNISTNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 代码中的128,64,10分别表示隐藏层和输出层的神经元数量
        # 隐藏层维度的基本原则
        # 是输入维度的1/2到1/4，输出层维度通常与类别数量相同
        # 一般是2的幂次方，便于内存对齐和并行化
        # 参数数量的计算
        # 输入层：28*28=784
        # 第一隐藏层：784*128 + 128 = 100480
        # 第二隐藏层：128*64 + 64 = 8256
        # 输出层：64*10 + 10 = 650
        # 总参数量：100480 + 8256 + 650 = 109386
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # view()：PyTorch中调整张量形状的方法，不改变数据内容，仅改变维度。
        # -1表示自动计算该维度的大小，28*28表示将输入展平为784维向量
        # 将四维张量 [64, 1, 28, 28] 转换为二维张量
        # 最终输出形状：
        # [64, 784]
        # view()与reshape()的区别
        # view()要求原张量的内存是连续的，如果不是，则会报错
        # reshape()可以处理非连续内存的张量，但可能会导致数据复制
        # 因此，view()通常更高效，但需要确保输入张量是连续的
        
        x = x.view(-1, 28 * 28)  # 展平输入
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x