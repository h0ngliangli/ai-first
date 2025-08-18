import argparse
import torch
import torchvision


def parse_args():
    argparser = argparse.ArgumentParser(description="练习使用FCN对FashionMNIST进行分类")
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="批大小, default %(default)d",
    )
    argparser.add_argument(
        "--seed", type=int, default=42, help="随机种子, default %(default)d"
    )
    argparser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据加载的子进程数, default %(default)d",
    )
    return argparser.parse_args()


def prepare_dataset():
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
    return train_dataset, val_dataset, test_dataset


def create_model():
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )
    return model


def train_model(args, model):
    epochs = 5
    batch_size = args.batch_size
    num_workers = args.num_workers
    seed = args.seed
    train_dataset, val_dataset, test_dataset = prepare_dataset()
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    torch.manual_seed(seed)
    for epoch in range(epochs):
        for images, labels in train_data_loader:
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
        accurate = eval_model(model, val_data_loader)
        print(f'eval accuracy: {accurate:.2f}')
    # test model
    accurate = eval_model(model, test_data_loader)
    print(f'final accuracy: {accurate:.2f}')


def eval_model(model, data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            output = model(images)
            total += labels.size(0)
            debug_1 = output.argmax(dim=1)
            debug_2 = labels
            debug_3 = output.argmax(dim=1) == labels
            debug_4 = (output.argmax(dim=1) == labels).sum()
            correct += (output.argmax(dim=1) == labels).sum().item()
        accuracy = correct / total
        return accuracy


if __name__ == "__main__":
    args = parse_args()
    model = create_model()
    train_model(args, model)
