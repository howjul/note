import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1：不仅保存了网络模型的机构，还保存了网络模型的参数
torch.save(vgg16, 'vgg16_method1.pth')

# 保存方式2：只保存模型参数（官方推荐），因为更加轻便小巧
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')
