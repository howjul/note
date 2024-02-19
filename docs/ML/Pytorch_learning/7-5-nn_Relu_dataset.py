import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./Dataset",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Howjul(nn.Module):
    def __init__(self):
        super(Howjul, self).__init__()
        self.relu1 = ReLU(inplace=False)  # 不进行原地替换
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        output = self.sigmoid1(x)
        return output


howjul = Howjul()

writer = SummaryWriter("logs_relu")
step = 0
for data in dataloader:
    imgs, labels = data
    writer.add_images("Input", imgs, step)
    output = howjul(imgs)
    writer.add_images("Output", output, step)
    step = step + 1
writer.close()
