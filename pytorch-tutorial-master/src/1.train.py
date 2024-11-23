# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *  # 导入自定义的模型（例如Tudui类定义）

# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

# 加载 CIFAR-10 数据集
train_data = torchvision.datasets.CIFAR10(
    root="../data", train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.CIFAR10(
    root="../data", train=False, transform=torchvision.transforms.ToTensor(), download=True
)

# 获取训练集和测试集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 打印训练集和测试集的大小
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 使用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建模型（Tudui是一个自定义的神经网络模型类）
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()  # 适用于分类任务的交叉熵损失函数

# 优化器：使用随机梯度下降（SGD）优化器
learning_rate = 1e-2  # 学习率为 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 训练过程的参数初始化
total_train_step = 0  # 训练步数
total_test_step = 0   # 测试步数
epoch = 10  # 总共训练的轮数

# 创建 TensorBoard SummaryWriter 实例，用于记录训练过程中的数据
writer = SummaryWriter("../logs_train")

# 开始训练
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    # 训练步骤
    tudui.train()  # 设置模型为训练模式
    for data in train_dataloader:
        imgs, targets = data  # 获取每一批数据（图像和标签）
        outputs = tudui(imgs)  # 使用模型进行前向传播
        loss = loss_fn(outputs, targets)  # 计算损失

        # 优化器优化模型
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 反向传播，计算当前梯度
        optimizer.step()  # 更新模型参数

        total_train_step += 1
        if total_train_step % 100 == 0:  # 每100次训练记录一次损失
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)  # 记录训练损失

    # 测试步骤
    tudui.eval()  # 设置模型为评估模式
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 测试时不需要计算梯度
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)  # 计算测试集损失
            total_test_loss += loss.item()

            accuracy = (outputs.argmax(1) == targets).sum()  # 计算正确的预测数量
            total_accuracy += accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)  # 记录测试损失
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)  # 记录测试准确率
    total_test_step += 1

    # 每一轮训练后保存模型
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

# 关闭 TensorBoard 的 writer
writer.close()
