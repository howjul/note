import torchvision
from torch import nn

# trained_data = torchvision.datasets.ImageNet("Dataset", split="train", download=True,
#                                              transform=torchvision.transforms.ToTensor())

# 加载模型
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print("ok")

# 加载数据集
dataset = torchvision.datasets.CIFAR10("./Dataset",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

# 第一种模型修改方法
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

# 第二种模型修改方法
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
