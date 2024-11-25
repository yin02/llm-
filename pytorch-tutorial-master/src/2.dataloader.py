import torchvision  # 导入 torchvision 库，用于加载标准数据集
from torch.utils.data import DataLoader  # 用于加载数据集并提供批量数据
from torch.utils.tensorboard import SummaryWriter  # 用于将训练过程中的信息写入 TensorBoard

# 1. 加载 CIFAR-10 测试数据集
# root：数据集存储的根目录；train=False：加载测试集；transform：对图像进行转换操作，这里使用 ToTensor() 转换图像为张量；
# download=True：如果数据集未下载，自动下载。
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

# 2. 使用 DataLoader 加载数据
# dataset：数据集对象，test_data 即为我们加载的 CIFAR-10 测试数据集；
# batch_size=64：每个批次包含的数据样本数；
# shuffle=True：是否打乱数据的顺序，避免模型过拟合；
# num_workers=0：数据加载时使用的子进程数，这里设置为 0，表示不使用子进程加载数据；
# drop_last=True：如果最后一个批次的样本数小于 batch_size，则丢弃该批次。
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 3. 获取测试数据集中的第一张图片及其对应的标签
# test_data[0] 返回第一张图片及其标签，img 为图像数据，target 为标签。
img, target = test_data[0]
print(img.shape)  # 打印该图片的形状，应该是 (3, 32, 32)，表示 CIFAR-10 图像为 32x32 的 RGB 图像
print(target)  # 打印该图片的标签，数字形式（0-9），表示图像的类别

# 4. 创建 TensorBoard 的 SummaryWriter，用于记录图像
# log_dir：TensorBoard 日志文件存储目录。此处为 "dataloader"，表示日志将存储在当前工作目录下的 "dataloader" 文件夹中。
writer = SummaryWriter("dataloader")

# 5. 循环训练 2 轮
for epoch in range(2):  # 训练 2 轮
    step = 0  # 每个 epoch 的步骤计数器
    # 6. 遍历测试集数据加载器
    for data in test_loader:
        imgs, targets = data  # 获取当前批次的图像和标签
        # print(imgs.shape)  # 打印每批数据的图像形状（通常是 (64, 3, 32, 32)，表示每批次 64 张 32x32 的 RGB 图像）
        # print(targets)  # 打印每批数据的标签，表示每张图像的分类标签

        # 7. 将当前批次的图像写入 TensorBoard
        # "Epoch: {}".format(epoch)：标记当前是第几个 epoch；
        # imgs：要记录的图像数据，应该是一个形状为 (N, C, H, W) 的四维张量；
        # step：当前步骤计数，用于区分每个批次。
        writer.add_images("Epoch: {}".format(epoch), imgs, step)

        step += 1  # 步骤计数器加 1

# 8. 关闭 TensorBoard 记录器
writer.close()
