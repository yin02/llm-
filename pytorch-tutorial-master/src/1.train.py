# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from model import *  # 导入自定义的模型（例如Tudui类定义）

# 1. 准备数据集
# 使用 torchvision 加载 CIFAR-10 数据集，自动下载并处理数据
train_data = torchvision.datasets.CIFAR10(
    root="../data", train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.CIFAR10(
    root="../data", train=False, transform=torchvision.transforms.ToTensor(), download=True
)

# 2. 获取数据集的大小
train_data_size = len(train_data)  # 训练集的大小
test_data_size = len(test_data)    # 测试集的大小

# 打印训练集和测试集的大小
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 3. 使用 DataLoader 加载数据集，批量大小为 64
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 4. 创建自定义的神经网络模型实例
tudui = Tudui()  # 假设 Tudui 是在 model.py 中定义的自定义模型类

# 5. 定义损失函数
loss_fn = nn.CrossEntropyLoss()  # 适用于分类任务的交叉熵损失函数

# 6. 定义优化器：使用随机梯度下降（SGD）优化器
learning_rate = 1e-2  # 学习率设为 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)  # 使用 SGD 优化器

# 7. 训练过程参数初始化
total_train_step = 0  # 训练步骤计数器
total_test_step = 0   # 测试步骤计数器
epoch = 10  # 总训练轮数设置为 10 轮

# 8. 创建 TensorBoard 的 SummaryWriter 实例，用于记录训练过程中的数据
writer = SummaryWriter("../logs_train")

# 9. 开始训练
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    # 训练步骤
    tudui.train()  # 设置模型为训练模式
    for data in train_dataloader:
        imgs, targets = data  # 从数据加载器中获取一批数据，imgs 是图像，targets 是标签
        outputs = tudui(imgs)  # 使用模型进行前向传播，输出模型预测值
        loss = loss_fn(outputs, targets)  # 计算当前批次的损失

        # 优化器优化模型
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 反向传播，计算当前梯度
        optimizer.step()  # 更新模型的权重和偏置

        total_train_step += 1  # 增加训练步骤计数
        if total_train_step % 100 == 0:  # 每100步记录一次训练损失
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))  # 打印当前训练步骤和损失
            writer.add_scalar("train_loss", loss.item(), total_train_step)  # 将训练损失记录到 TensorBoard

    # 10. 测试步骤
    tudui.eval()  # 设置模型为评估模式，禁用 dropout 和 batch normalization
    total_test_loss = 0  # 初始化测试集总损失
    total_accuracy = 0  # 初始化正确的预测总数
    with torch.no_grad():  # 测试时不计算梯度，节省内存和计算资源
        for data in test_dataloader:
            imgs, targets = data  # 获取测试集中的一批数据
            outputs = tudui(imgs)  # 使用模型进行前向传播
            loss = loss_fn(outputs, targets)  # 计算损失
            total_test_loss += loss.item()  # 累加测试集损失

            # 计算当前批次的准确率
            accuracy = (outputs.argmax(1) == targets).sum()  # 输出类别的索引与目标标签对比，统计正确预测的数量
            total_accuracy += accuracy  # 累加正确预测的数量

    # 11. 输出测试集的总体损失和准确率
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))  # 测试集的准确率
    writer.add_scalar("test_loss", total_test_loss, total_test_step)  # 记录测试损失到 TensorBoard
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)  # 记录测试准确率到 TensorBoard
    total_test_step += 1  # 增加测试步骤计数

    # 12. 每一轮训练后保存模型
    torch.save(tudui, "tudui_{}.pth".format(i))  # 保存模型，文件名包含当前轮数
    print("模型已保存")

# 13. 关闭 TensorBoard 的 writer
writer.close()
