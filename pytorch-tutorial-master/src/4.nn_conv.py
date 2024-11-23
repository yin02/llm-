# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的功能性函数库，包含卷积操作等

# 1. 定义输入张量（5x5的二维张量）
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 2. 定义卷积核（3x3的二维卷积核）
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 3. 调整输入和卷积核的形状，符合 `conv2d` 函数的输入要求
# F.conv2d 的输入需要是形状为 (batch_size, channels, height, width) 的四维张量
input = torch.reshape(input, (1, 1, 5, 5))  # 转换为形状 (1, 1, 5, 5)，即1个batch，1个通道，5x5的图像
kernel = torch.reshape(kernel, (1, 1, 3, 3))  # 转换为形状 (1, 1, 3, 3)，即1个通道，3x3的卷积核

print(input.shape)  # 输出输入张量的形状
print(kernel.shape)  # 输出卷积核的形状

# 4. 使用 F.conv2d 执行二维卷积操作，步长 (stride)=1
output = F.conv2d(input, kernel, stride=1)
print(output)  # 打印卷积结果

# 5. 使用 F.conv2d 执行二维卷积操作，步长 (stride)=2
output2 = F.conv2d(input, kernel, stride=2)
print(output2)  # 打印卷积结果

# 6. 使用 F.conv2d 执行二维卷积操作，步长 (stride)=1，填充 (padding)=1
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)  # 打印卷积结果
