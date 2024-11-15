import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10("./Dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./Dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度: {}".format(train_data_size))
print("测试数据集长度: {}".format(test_data_size))

# 利用 dataloader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
howjul = Howjul()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(howjul.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10

# 添加 tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    howjul.train()
    for data in train_dataloader:
        imgs, labels = data
        outputs = howjul(imgs)
        loss = loss_fn(outputs, labels)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    howjul.eval()
    total_test_loss = 0
    total_accuracy = 0
    # 表示 with 里面的代码没有梯度，保证不会进行参数优化
    with torch.no_grad():
        for data in test_dataloader:
            imgs, labels = data
            outputs = howjul(imgs)
            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item()  # item()可以将一个类型转化为数字
            accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy += accuracy.item()

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(howjul, "howjul_{}.pth".format(i))
    # torch.save(howjul.state_dict(), "howjul_{}.pth".format(i))
    print("模型已保存")

writer.close()
