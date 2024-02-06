"""
DataSet 类实战 1 - 加载数据集
数据集的标注方式为 文件的名字就是标注，文件夹的名字就是标注

当数据集要存储的信息较多，比如一张图片的位置，标签，大小等等，就需要自定义数据集类
一般采用再另一个文件夹中建立同名的 txt 文件，里面存储的是图片的位置和标签
"""
from torch.utils.data import Dataset
# utils 是工具包，data 是数据包，Dataset 是数据集类
from PIL import Image
# PIL 是 Python 图像处理库
import os


class MyData(Dataset):

    def __init__(self, root_dir, img_dir, label_dir):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_path = os.path.join(self.root_dir, self.img_dir)  # 地址拼接
        self.img_set = os.listdir(self.img_path)  # 列出文件夹下所有文件的名称
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.label_set = os.listdir(self.label_path)

    def __getitem__(self, idx):
        img_name = self.img_set[idx]  # 获取对应路径下的文件名
        label_name = self.label_set[idx]  # 拼接得到路径
        img_item_path = os.path.join(self.img_path, img_name)  # 读取图片
        label_item_path = os.path.join(self.label_path, label_name)
        img = Image.open(img_item_path)
        # 读取 label.txt 中的内容
        label = open(label_item_path).read().strip()
        return img, label

    def __len__(self):
        return len(self.img_path)


root = 'Dataset/train'
ants_img_dir = 'ants_image'
ants_label_dir = 'ants_label'
bees_img_dir = 'bees_image'
bees_label_dir = 'bees_label'
ants_dataset = MyData(root, ants_img_dir, ants_label_dir)
bees_dataset = MyData(root, bees_img_dir, bees_label_dir)

# train_dataset = ants_dataset + bees_dataset  # 拼接数据集

# img, label = ants_dataset[0]
# img.show() # 会显示出第一张 ants 数据集的图片
