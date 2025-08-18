# 使用Sequential构建神经网络，输入3个特征，输出1个值，
# 通过学习实现Max Index的神经网络
import torch
import torch.utils.data as data

# 构建神经网络模型
# 输出3个值表示3个位置的概率，而不是1个值

# 自定义神经网络module，返回输入的最大值

model = torch.nn.Sequential(
    torch.nn.Linear(3, 128),  # 第一层：3输入 -> 128隐藏单元
    torch.nn.ReLU(),  # 激活函数
    torch.nn.Linear(128, 64),  # 第二层：128 -> 64
    torch.nn.ReLU(),  # 激活函数
    torch.nn.Linear(64, 3),  # 输出层：64 -> 3（3个位置的得分）
)

# 损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()  # 交叉熵损失，适合分类问题
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 生成训练数据
learning_size = 5000
input_data = torch.randn(learning_size, 3)  # 随机生成输入数据
target_indices = input_data.argmax(dim=1)  # 获取最大值的索引作为标签

# 创建数据集和数据加载器
dataset = data.TensorDataset(input_data, target_indices)
dataloader = data.DataLoader(dataset, batch_size=100, shuffle=True)

print("开始训练...")
print(f"训练数据大小: {learning_size}")
print(f"网络结构: {model}")

# 训练循环
for epoch in range(50):
    total_loss = 0
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)  # 前向传播
        loss = loss_fn(outputs, batch_targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Average Loss: {total_loss/len(dataloader):.4f}")

print("训练完成！")

# 测试模型
print("\n=== 测试模型 ===")
test_cases = [
    [1.0, 2.0, 3.0],  # 最大值在位置2
    [5.0, 1.0, 2.0],  # 最大值在位置0
    [2.0, 8.0, 1.0],  # 最大值在位置1
    [-1.0, -2.0, 0.5],  # 最大值在位置2
    [3.5, 3.5, 2.0],  # 最大值在位置0或1
]

with torch.no_grad():
    for i, test_input in enumerate(test_cases):
        input_tensor = torch.tensor([test_input], dtype=torch.float32)
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        actual_idx = torch.argmax(input_tensor, dim=1).item()

        print(f"测试 {i+1}: 输入 {test_input}")
        print(f"  实际最大值位置: {actual_idx}")
        print(f"  预测最大值位置: {predicted_idx}")
        print(f"  预测正确: {'✓' if predicted_idx == actual_idx else '✗'}")
        print(f"  网络输出概率: {torch.softmax(output, dim=1).squeeze().tolist()}")
        print()
