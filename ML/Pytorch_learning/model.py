import torch
from torch import nn


# 搭建神经网络
class Howjul(nn.Module):
    def __init__(self):
        super(Howjul, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    howjul = Howjul()
    # 指定 input 为64个样本3通道，长和宽为32x32
    input = torch.ones((64, 3, 32, 32))
    output = howjul(input)
    # 输出应为[64, 10]，64为样本数量，10为类别数目
    print(output.shape)
