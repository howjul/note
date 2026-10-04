import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./Dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Howjul(nn.Module):
    def __init__(self):
        super(Howjul,self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x


howjul = Howjul()

for imgs, labels in dataloader:
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)  # 把输入展成一行
    print(output.shape)
    output = howjul(output)
    print(output.shape)
    print("\n")

