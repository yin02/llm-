# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torch  # 导入 PyTorch 库，用于构建神经网络和进行张量操作
import torchvision  # 导入 torchvision 库，用于加载标准数据集，如 CIFAR-10
from torch import nn  # 从 PyTorch 中导入神经网络模块（nn），用来构建神经网络
from torch.nn import Linear  # 导入线性层（全连接层），用于做线性变换
from torch.utils.data import DataLoader  # 用于加载数据集的 DataLoader，可以按批次加载数据

# 1. 加载 CIFAR-10 测试数据集
# CIFAR-10 数据集包含 10 类的 32x32 像素图像。train=False 表示加载的是测试集。
# transform=torchvision.transforms.ToTensor()：将图像转换为 PyTorch 张量格式，方便网络处理。
# download=True 表示如果数据集尚未下载，它会自动下载。
dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 2. 使用 DataLoader 来按批次加载数据
# batch_size=64 表示每次加载 64 张图像，drop_last=True 表示如果最后一个批次不足 64 张图片，则丢弃。
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

# 3. 定义自定义的神经网络模型 Tudui，继承自 nn.Module
# 神经网络模型包含一个线性层，将 196608 维输入映射到 10 维输出。
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()  # 调用父类的构造函数，初始化神经网络模块
        # 定义一个线性层，输入特征数量为 196608，输出特征数量为 10
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        # 定义前向传播过程，将输入通过线性层进行处理
        output = self.linear1(input)
        return output

# 4. 创建神经网络模型实例
tudui = Tudui()  # 实例化自定义的 Tudui 模型

# 5. 遍历数据加载器中的每个批次
# DataLoader 会按批次返回图像和标签数据
for data in dataloader:
    imgs, targets = data  # imgs 是输入图像，targets 是图像对应的标签
    print(imgs.shape)  # 打印当前批次图像的形状，应该是 (batch_size, 3, 32, 32)
    
    # 6. 对图像进行展平操作，将每张图像从 3x32x32 转换为 1D 向量
    # CIFAR-10 图像的尺寸是 32x32 像素，3 个通道，展平后的形状是 (batch_size, 196608)
    output = torch.flatten(imgs)  # 将每张 3x32x32 的图像展平为一维向量

    print(output.shape)  # 打印展平后的张量形状，应该是 (batch_size, 196608)
    
    # 7. 将展平后的图像输入到神经网络中，得到输出
    output = tudui(output)  # 通过 Tudui 模型进行前向传播
    print(output.shape)  # 打印模型的输出形状，应该是 (batch_size, 10)，表示 10 个类别的预测值

