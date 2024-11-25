# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torch  # 导入 PyTorch 库
import torchvision  # 导入 torchvision 库，用于加载标准数据集
from torch import nn  # 导入 PyTorch 的神经网络模块
from torch.nn import ReLU, Sigmoid  # 导入 ReLU 和 Sigmoid 激活函数
from torch.utils.data import DataLoader  # 用于加载数据集的 DataLoader
from torch.utils.tensorboard import SummaryWriter  # 用于记录训练过程的数据到 TensorBoard

# 1. 定义输入张量并调整形状
input = torch.tensor([[1, -0.5],
                      [-1, 3]])

# 2. 调整张量的形状，转换为四维张量，符合卷积操作的输入要求
input = torch.reshape(input, (-1, 1, 2, 2))  # 调整形状为 (batch_size, channels, height, width)
print(input.shape)  # 输出调整后的张量形状

# 3. 加载 CIFAR-10 测试数据集
dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

# 4. 使用 DataLoader 加载数据，批次大小为 64
dataloader = DataLoader(dataset, batch_size=64)

# 5. 定义自定义的神经网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 定义两个激活函数：ReLU 和 Sigmoid
        self.relu1 = ReLU()  # ReLU 激活函数
        self.sigmoid1 = Sigmoid()  # Sigmoid 激活函数

    def forward(self, input):
        # 在前向传播过程中，使用 Sigmoid 激活函数对输入进行处理
        output = self.sigmoid1(input)
        return output

# 6. 创建模型对象
tudui = Tudui()

# 7. 创建 TensorBoard 记录器
writer = SummaryWriter("../logs_relu")

# 8. 遍历数据加载器，并记录图像数据到 TensorBoard
step = 0  # 步骤计数器，用于区分不同的批次
for data in dataloader:
    imgs, targets = data  # 获取每个批次的图像和标签
    # 将原始输入图像记录到 TensorBoard
    writer.add_images("input", imgs, global_step=step)
    
    # 将每个批次的图像输入模型，得到输出
    output = tudui(imgs)
    # 将模型输出的图像记录到 TensorBoard
    writer.add_images("output", output, step)
    
    step += 1  # 步骤计数器加 1

# 9. 关闭 TensorBoard 记录器
writer.close()
