from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("Dataset/train/ants_image/0013035.jpg")
print(img)

# ToTensor 的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize 的使用
# Normalize a tensor image with mean and standard deviation.
# ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# Resize 的使用
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)  # 输出的还是PIL图片类型
img_resize = trans_totensor(img_resize)  # 把PIL图片转化回tensor数据类型
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop 的使用
trans_random = transforms.RandomCrop(512)
# or 指定 H, W
# trans_random = transforms.RandomCrop((512, 512))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("Random Crop", img_crop, i)

writer.close()
