# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torch  # 导入 PyTorch 库
import torchvision  # 导入 torchvision 库，用于加载标准数据集
from torch import nn  # 导入 PyTorch 的神经网络模块
from torch.nn import Conv2d  # 导入卷积层类 Conv2d
from torch.utils.data import DataLoader  # 用于加载数据集的 DataLoader
from torch.utils.tensorboard import SummaryWriter  # 用于记录训练过程的数据到 TensorBoard

# 1. 加载 CIFAR-10 测试数据集
# root="../data"：数据集存储路径；train=False：加载测试集；transform=torchvision.transforms.ToTensor()：将图像转换为 Tensor；
# download=True：如果数据集未下载，自动下载
dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 2. 创建数据加载器
# DataLoader 用于按批次加载数据，batch_size=64：每个批次加载 64 张图片
dataloader = DataLoader(dataset, batch_size=64)

# 3. 自定义神经网络模型 Tudui，继承自 nn.Module
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 定义一个 3 通道输入，6 通道输出的卷积层，卷积核大小为 3x3，步长为 1，不使用填充
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        # 在前向传播中，通过卷积层处理输入
        x = self.conv1(x)
        return x

# 4. 创建模型对象
tudui = Tudui()

# 5. 创建 TensorBoard 记录器，用于记录输入和输出图像
writer = SummaryWriter("../logs")

# 6. 训练循环（这里只是演示卷积操作，不涉及实际训练）
step = 0  # 步骤计数器，用于区分不同的批次
for data in dataloader:
    imgs, targets = data  # 获取每个批次的图像和标签
    output = tudui(imgs)  # 通过模型进行前向传播，得到卷积输出
    print(imgs.shape)  # 打印输入图像的形状
    print(output.shape)  # 打印卷积后的输出形状

    # 7. 将输入图像记录到 TensorBoard 中
    writer.add_images("input", imgs, step)

    # 8. 对卷积后的输出进行形状调整
    output = torch.reshape(output, (-1, 3, 30, 30))  # 由于卷积核的设置，输出形状为 (batch_size, 6, 30, 30)，调整为 (batch_size, 3, 30, 30)

    # 9. 将卷积后的输出图像记录到 TensorBoard 中
    writer.add_images("output", output, step)

    # 10. 步骤计数器增加
    step = step + 1

# 11. 关闭 TensorBoard 记录器
writer.close()
