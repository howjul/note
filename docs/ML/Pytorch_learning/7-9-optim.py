import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./Dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)


class Howjul(nn.Module):
    def __init__(self):
        super(Howjul, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
howjul = Howjul()
optim = torch.optim.SGD(howjul.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, labels = data
        output = howjul(imgs)
        result_loss = loss(output, labels)
        optim.zero_grad()  # 清除梯度信息
        result_loss.backward()  # 计算出梯度
        optim.step()  # 对每个参数进行优化
        running_loss += result_loss
    print("Epoch: {}/{} Loss: {}".format(epoch+1, 20, running_loss))
