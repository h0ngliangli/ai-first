# 配置神经网络使得第一层接受三个输入，第二层有一个输出，
# 输出的是上一层的最大值的位置。 比如[1,2,3]，输出是2。
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.argmax

class MaxIndexNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(3, 1)  # 第一层接受三个输入，输出一个值

    def forward(self, x):
        # x = self.fc1(x)  # 线性变换
        return x.argmax(dim=1)  # 返回最大值的位置
    
# 测试神经网络
if __name__ == "__main__":
    model = MaxIndexNN()
    # 输入三个数，输出最大值的位置
    input_data = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    output = model(input_data)
    print("Input:", input_data)
    print("Output (Max Index):", output)
    print(output == torch.tensor([2, 0]))  # 验证输出是否正确