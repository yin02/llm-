from torch.utils.data import Dataset, DataLoader  # 导入 PyTorch 中用于数据加载的库
import numpy as np  # 用于处理数据（本代码中没用到 numpy，可能是准备处理数组）
from PIL import Image  # 用于处理图像
import os  # 用于操作系统相关的路径操作
from torchvision import transforms  # 用于图像数据预处理
from torch.utils.tensorboard import SummaryWriter  # 用于将数据写入 TensorBoard
from torchvision.utils import make_grid  # 用于将图像合并成网格展示

writer = SummaryWriter("logs")  # 创建一个 TensorBoard 记录器，保存日志到 logs 目录

# 自定义数据集类 MyData，继承自 Dataset 类
class MyData(Dataset):
    def __init__(self, root_dir, image_dir, label_dir, transform):
        """
        初始化数据集类
        :param root_dir: 数据集根目录
        :param image_dir: 图像文件夹的子目录
        :param label_dir: 标签文件夹的子目录
        :param transform: 用于数据增强的转换操作
        """
        self.root_dir = root_dir  # 根目录路径
        self.image_dir = image_dir  # 图像文件夹路径
        self.label_dir = label_dir  # 标签文件夹路径
        self.label_path = os.path.join(self.root_dir, self.label_dir)  # 标签文件的完整路径
        self.image_path = os.path.join(self.root_dir, self.image_dir)  # 图像文件的完整路径

        # 获取目录下的所有文件名列表
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        
        self.transform = transform  # 用于图像处理的转换（如大小调整、转为张量等）
        
        # 由于图像和标签文件名相同，按照文件名排序，以保证图像与标签一一对应
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        :param idx: 数据集中的索引
        :return: 一个包含图像和标签的字典
        """
        # 根据索引获得图像和标签的文件名
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        
        # 获取图像和标签文件的完整路径
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        
        # 打开图像文件
        img = Image.open(img_item_path)
        
        # 读取标签文件，标签文件应该是一个文本文件，每行包含一个标签
        with open(label_item_path, 'r') as f:
            label = f.readline().strip()  # 读取标签，并去除换行符
        
        # 对图像进行转换（例如调整大小和转为 Tensor）
        img = self.transform(img)
        
        # 将图像和标签封装为字典返回
        sample = {'img': img, 'label': label}
        return sample

    def __len__(self):
        """
        返回数据集的大小（即样本数量）
        """
        assert len(self.image_list) == len(self.label_list)  # 确保图像和标签的数量相等
        return len(self.image_list)

# 主程序入口
if __name__ == '__main__':
    # 定义图像转换操作，调整图像大小为 256x256，并转为 Tensor
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    # 数据集路径和文件夹名称
    root_dir = "dataset/train"
    image_ants = "ants_image"  # 蚂蚁图像文件夹
    label_ants = "ants_label"  # 蚂蚁标签文件夹
    
    # 创建蚂蚁数据集
    ants_dataset = MyData(root_dir, image_ants, label_ants, transform)
    
    # 蜂蜜数据集路径
    image_bees = "bees_image"  # 蜂蜜图像文件夹
    label_bees = "bees_label"  # 蜂蜜标签文件夹
    
    # 创建蜂蜜数据集
    bees_dataset = MyData(root_dir, image_bees, label_bees, transform)
    
    # 将两个数据集合并
    train_dataset = ants_dataset + bees_dataset

    # 创建 DataLoader 用于批量加载数据
    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)

    # 将第 119 个样本的图像写入 TensorBoard
    writer.add_image('error', train_dataset[119]['img'])
    writer.close()
    
    # 下面是一个数据加载循环的代码示例（暂时注释掉）
    # for i, j in enumerate(dataloader):
    #     print(type(j))  # 打印每个 batch 的类型
    #     print(i, j['img'].shape)  # 打印每个 batch 中图像的形状
    #     writer.add_image("train_data_b2", make_grid(j['img']), i)  # 将每个 batch 的图像加入 TensorBoard
    #
    # writer.close()
