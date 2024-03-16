import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
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
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


howjul = Howjul()

writer = SummaryWriter("logs_maxpool")
step = 0

for data in dataloader:
    imgs, labels = data
    writer.add_images("Input", imgs, step)
    output = howjul(imgs)
    writer.add_images("Output", output, step)
    step = step + 1

writer.close()
