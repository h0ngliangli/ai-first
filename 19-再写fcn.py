# 目的：练习使用FCN对FashionMNIST数据集进行分类
# 改进：
# 1. 去掉最后一层LogSoftmax输出，因为损失函数是CrossEntropyLoss已经包含了softmax
# 2. datasets中的download设为True. 数据集只需要下载一次
# 3. 使用官方推荐的均值和标准差 (0.2860, 0.3530)
# 4. 评估指标只显示了准确率，输出loss
# 5. 使用model.to(device)， images.to(device) 将数据移动到GPU
# 6. 代码结构： 将训练和测试代码封装成函数
# 7. 添加模型保存和加载功能
# 8. 从训练集中划分一部分作为验证集，在训练过程中进行验证，可以及时发现模型是否在训练集上过拟合。
# 9. 增加早停机制：当验证集性能不再提升时，停止训练
# 10. 通过num_workers参数来加速数据加载和CPU利用率
# 11. 设置随机种子，以确保结果可复现
# 12. 加入学习率衰减，比如torch.optim.lr_scheduler.StepLR
# 13. 使用tqdm包显示训练进度 (conda install tqdm)
#     tqdm是一个显示进度条的Python库
# 14. 用argparse解析命令行参数
import argparse
import torch
import torchvision
import numpy
import random
import tqdm
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="Train a FCN on FashionMNIST")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--num_workers", type=int, default=4
)  # 数据加载(DataLoader)的子进程数，不影响模型计算。
args = parser.parse_args()
batch_size = args.batch_size
# batch_size变大，epoch需要的batch数变小，数据加载次数变少，整体训练时间变短，但
# 可能导致内存占用增加, 以及泛化能力下降（大batch使模型更容易收敛，而锁死在局部最优解，
# 忽略了数据的多样性）
seed = args.seed  # 设置随机种子, 一般只要种子固定，结果就可复现
num_workers = args.num_workers  # 数据加载(DataLoader)的子进程数，不影响模型计算。

torch.manual_seed(seed)
numpy.random.seed(seed)
random.seed(seed)

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.286,), (0.353,)),
    ]
)
train_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=True, download=True, transform=transforms
)
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [50000, 10000]
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transforms
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(28 * 28, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)
model_path = "1-fcn.pth"
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 学习率衰减，每10个epoch将学习率降低一半 （这里用不到，因为epoch大概在8的时候就结束了）
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
patience = 3


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    device,
    epochs,
    save_path=model_path,
):
    min_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm.tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # 前向传播 outputs记录了完整的计算图
            outputs = model(images)
            # 计算损失 loss
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        training_loss = running_loss / len(train_dataloader)
        # 验证集评估
        model.eval()
        val_loss, val_accuracy, cm = evaluate_model(
            model, val_dataloader, loss_fn, device
        )

        # 早停机制
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
            try:
                torch.save(model.state_dict(), save_path)
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            patience_counter += 1
            # Optionally reload the best model so far (early stopping)
            try:
                model.load_state_dict(torch.load(save_path, weights_only=True))
            except Exception as e:
                print(f"Error loading model: {e}")
        print(
            f"Epoch {epoch + 1} "
            f"loss: {training_loss:.2f} "
            f"loss(val): {val_loss:.2f} "
            f"acc(val): {val_accuracy * 100:.2f}% "
            f"patience: {patience_counter}/{patience} "
        )
        scheduler.step()
        if patience_counter >= patience:
            break


def evaluate_model(model, test_dataloader, loss_fn, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        correct = 0
        total = 0
        running_loss = 0.0
        for images, labels in tqdm.tqdm(test_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            # max()中dim=1表示返回每行的最大值及其索引
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        avg_loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        cm = confusion_matrix(all_labels, all_preds)
        return (avg_loss, accuracy, cm)


print(f"Training on {device}")
train_model(
    model, train_dataloader, val_dataloader, loss_fn, optimizer, device, epochs=50
)
print(f"Evaluating on {device}")
try:
    model.load_state_dict(torch.load(model_path, weights_only=True))
except Exception as e:
    print(f"Error loading model: {e}")
test_loss, test_accuracy, cm = evaluate_model(model, test_dataloader, loss_fn, device)
print(f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(cm)
