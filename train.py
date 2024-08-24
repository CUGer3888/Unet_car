# import torch
# import torch.nn as nn
# import torch.optim as optim
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from tqdm import tqdm
# from model import Unet
#
# from utils import getloaders
#
# import numpy as np
# import random
#
# LEARNING_RATE = 1e-8
# BATCH_SIZE = 8
# epochs = 100
# NUM_WORKERS = 2
# PIN_MEMORY = True
#
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # print(DEVICE)
#
# TRAIN_IMG_DIR = "./deta/train_img_dir"
# TRAIN_MASK_DIR = "./deta/train_mask_dir"
# VAL_IMG_DIR = "./deta/val_img_dir"
# VAL_MASK_DIR = "./deta/val_mask_dir"
#
# IMAGE_HEIGHT = 160
# IMAGE_WIDTH = 240
#
# train_losses = []
# val_acc = []
# val_dice = []
#
# # 设置随机种子
# # 生成一个1到100之间的随机数作为种子
# seed = random.randint(1, 100)
# # 设置CPU的随机种子
# torch.manual_seed(seed)
# # 设置GPU的随机种子
# torch.cuda.manual_seed(seed)
# # 设置所有GPU的随机种子
# torch.cuda.manual_seed_all(seed)
# # 设置numpy的随机种子
# np.random.seed(seed)
# # 设置Python的随机种子
# random.seed(seed)
# # 设置cuDNN的确定性模式
# torch.backends.cudnn.deterministic = True
# # 设置cuDNN的基准模式
# torch.backends.cudnn.benchmark = False
#
#
# def train_fn(loader, model, loss_fn, optimizer, scaler):
#     loop = tqdm(loader)
#     total_loss = 0.0
#     for batch_idx, (data, targets) in enumerate(loop):
#         data = data.to(DEVICE)
#         targets = targets.unsqueeze(1).float().to(DEVICE)
#
#         with torch.cuda.amp.autocast():
#             preds = model(data)
#             loss = loss_fn(preds, targets)
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#
#         total_loss += loss.item()
#
#         loop.set_postfix(loss=loss.item())
#     return total_loss / len(loader)
#
#
# def cheack_accuracy(loader, model, DEVICE='cuda'):
#     num_correct = 0
#     num_pixels = 0
#     dice_socre = 0
#     model.eval()
#
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(DEVICE)
#             y = y.unsqueeze(1).to(DEVICE)
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#             num_correct += (preds == y).sum()
#             num_pixels += torch.numel(preds)
#             dice_socre += (2 * (preds * y).sum()) / (preds.sum() + y.sum() + 1e-6)
#
#     accuracy = round(float(num_correct / num_pixels), 4)
#     dice = round(float(dice_socre / len(loader)), 4)
#     print(f'Got{num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}')
#     print(f'Dice Score:{dice_socre / len(loader)}')
#
#     model.train()
#     return accuracy, dice
#
#
# def main():
#     train_transform = A.Compose(
#         [
#             A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
#             A.Rotate(limit=35, p=1.0),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.Normalize(
#                 mean=[0.0, 0.0, 0.0],
#                 std=[1.0, 1.0, 1.0],
#                 max_pixel_value=255.0,
#             ),
#             ToTensorV2(),
#         ],
#     )
#     val_transform = A.Compose(
#         [
#             A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
#             A.Normalize(
#                 mean=[0.0, 0.0, 0.0],
#                 std=[1.0, 1.0, 1.0],
#                 max_pixel_value=255.0,
#             ),
#             ToTensorV2(),
#         ],
#     )
#     train_loader, val_loader = getloaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
#                                           train_transform,
#                                           val_transform,
#                                           BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
#                                           )
#     model = Unet(in_channels=3, out_channels=1).to(device=DEVICE)
#     loss_fn = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     scaler = torch.cuda.amp.GradScaler()
#
#     for index in range(epochs):
#         print("Current Epoch:", index)
#         train_loss = train_fn(train_loader, model, loss_fn, optimizer, scaler)
#         train_losses.append(train_loss)
#
#         accuracy, dice = cheack_accuracy(val_loader, model, DEVICE=DEVICE)
#         val_acc.append(accuracy)
#         val_dice.append(dice)
#     # 保存模型
#     save_path = str(epochs) + 'model.pth'
#     torch.save(model.state_dict(), save_path)
#
#
# if __name__ == "__main__":
#     main()
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import Unet
import time
from utils import getloaders

import numpy as np
import random

# 设置学习率
LEARNING_RATE = 1e-8
# 设置批量大小
BATCH_SIZE = 8
# 设置训练轮数
epochs = 100
# 设置工作线程数
NUM_WORKERS = 2
# 设置是否使用锁页内存
PIN_MEMORY = True

# 设置设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(DEVICE)

# 设置训练集和验证集的图片和掩码路径
TRAIN_IMG_DIR = "./deta/train_img_dir"
TRAIN_MASK_DIR = "./deta/train_mask_dir"
VAL_IMG_DIR = "./deta/val_img_dir"
VAL_MASK_DIR = "./deta/val_mask_dir"

# 设置图片的高度和宽度
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

# 初始化训练损失和验证准确率
train_losses = []
val_acc = []
val_dice = []

# 设置随机种子
# 生成一个1到100之间的随机数作为种子
seed = random.randint(1, 100)
# 设置CPU的随机种子
torch.manual_seed(seed)
# 设置GPU的随机种子
torch.cuda.manual_seed(seed)
# 设置所有GPU的随机种子
torch.cuda.manual_seed_all(seed)
# 设置numpy的随机种子
np.random.seed(seed)
# 设置Python的随机种子
random.seed(seed)
# 设置cuDNN的确定性模式
torch.backends.cudnn.deterministic = True
# 设置cuDNN的基准模式
torch.backends.cudnn.benchmark = False


def train_fn(loader, model, loss_fn, optimizer, scaler):
    # 设置进度条
    loop = tqdm(loader)
    # 初始化总损失
    total_loss = 0.0
    # 遍历数据集
    for batch_idx, (data, targets) in enumerate(loop):
        # 将数据移动到设备上
        data = data.to(DEVICE)
        # 将标签扩展为1维，并转换为浮点型，并移动到设备上
        targets = targets.unsqueeze(1).float().to(DEVICE)

        # 使用自动混合精度
        with torch.cuda.amp.autocast():
            # 前向传播
            preds = model(data)
            # 计算损失
            loss = loss_fn(preds, targets)
        # 梯度清零
        optimizer.zero_grad()
        # 使用自动混合精度进行反向传播
        scaler.scale(loss).backward()
        # 使用自动混合精度进行优化
        scaler.step(optimizer)
        # 更新自动混合精度
        scaler.update()

        # 累加损失
        total_loss += loss.item()

        # 更新进度条
        loop.set_postfix(loss=loss.item())
    # 返回平均损失
    return total_loss / len(loader)


def cheack_accuracy(loader, model, DEVICE='cuda'):
    # 初始化正确像素数和总像素数
    num_correct = 0
    num_pixels = 0
    # 初始化Dice分数
    dice_socre = 0
    # 设置模型为评估模式
    model.eval()

    # 不计算梯度
    with torch.no_grad():
        # 遍历数据集
        for x, y in loader:
            # 将数据移动到设备上
            x = x.to(DEVICE)
            # 将标签移动到设备上
            y = y.unsqueeze(1).to(DEVICE)
            # 前向传播
            preds = torch.sigmoid(model(x))
            # 将预测结果转换为二值型
            preds = (preds > 0.5).float()
            # 累加正确像素数
            num_correct += (preds == y).sum()
            # 累加总像素数
            num_pixels += torch.numel(preds)
            # 累加Dice分数
            dice_socre += (2 * (preds * y).sum()) / (preds.sum() + y.sum() + 1e-6)

    # 计算准确率
    accuracy = round(float(num_correct / num_pixels), 4)
    # 计算Dice分数
    dice = round(float(dice_socre / len(loader)), 4)
    # 打印结果
    print(f'Got{num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}')
    print(f'Dice Score:{dice_socre / len(loader)}')

    # 设置模型为训练模式
    model.train()
    # 返回准确率和Dice分数
    return accuracy, dice


def main():
    # 设置训练集的变换
    train_transform = A.Compose(
        [
            # 调整图片大小
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # 旋转图片
            A.Rotate(limit=35, p=1.0),
            # 水平翻转图片
            A.HorizontalFlip(p=0.5),
            # 垂直翻转图片
            A.VerticalFlip(p=0.5),
            # 标准化图片
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            # 将图片转换为张量
            ToTensorV2(),
        ],
    )
    # 设置验证集的变换
    val_transform = A.Compose(
        [
            # 调整图片大小
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # 标准化图片
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            # 将图片转换为张量
            ToTensorV2(),
        ],
    )
    # 获取训练集和验证集的加载器
    train_loader, val_loader = getloaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
                                          train_transform,
                                          val_transform,
                                          BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
                                          )
    # 初始化模型
    model = Unet(in_channels=3, out_channels=1).to(device=DEVICE)
    # 初始化损失函数
    loss_fn = nn.BCEWithLogitsLoss()
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 初始化自动混合精度
    scaler = torch.cuda.amp.GradScaler()

    # 遍历训练轮数
    for index in range(epochs):
        # 打印当前轮数
        print("Current Epoch:", index)
        # 训练模型
        train_loss = train_fn(train_loader, model, loss_fn, optimizer, scaler)
        # 将训练损失添加到列表中
        train_losses.append(train_loss)

        # 验证模型
        accuracy, dice = cheack_accuracy(val_loader, model, DEVICE=DEVICE)
        # 将验证准确率和Dice分数添加到列表中
        val_acc.append(accuracy)
        val_dice.append(dice)
    # 保存模型
    #获得今天的时间日期
    time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_path = str(time) + 'model.pth'
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
