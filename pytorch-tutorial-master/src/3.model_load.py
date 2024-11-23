# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torch  # 导入 PyTorch 库
from model_save import *  # 假设从 'model_save' 文件中导入一些必要的内容

# 方式1 - 保存和加载模型
# 使用 torch.load() 直接加载保存的模型
model = torch.load("vgg16_method1.pth")
# print(model)  # 如果需要查看加载的模型内容，可以取消注释

# 方式2 - 加载模型，推荐的方式
# 创建一个 VGG16 模型对象，但不加载预训练权重（pretrained=False）
vgg16 = torchvision.models.vgg16(pretrained=False)
# 加载模型的权重
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")  # 这种方式会加载整个模型，包括架构和权重
# print(vgg16)  # 如果需要查看模型架构，可以取消注释

# 陷阱1：模型定义与加载
# 假设我们自定义了一个名为 Tudui 的模型类，包含一个卷积层
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)  # 定义了一个 3 通道输入，64 通道输出的卷积层
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x
#
# 1. 使用 torch.load() 加载模型：
model = torch.load('tudui_method1.pth')
print(model)  # 打印加载的模型，查看模型的具体结构和参数
