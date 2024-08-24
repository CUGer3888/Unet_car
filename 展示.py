
import numpy as np
import torch
import cv2
from model import Unet
from torchvision import transforms
from PIL import Image
import os

# 预处理
transform = transforms.Compose([
    transforms.Resize((160, 240)),  # 缩放图像
    transforms.ToTensor(),
])
# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Unet(in_channels=3, out_channels=1).to(device=device)
net.load_state_dict(torch.load('./model.pth', map_location=device))
net.to(device)
# 测试模式
net.eval()

pic_path = "C:\\Users\\CUGac\\PycharmProjects\\astar\\.venv\\Scripts\\U-Net\\test\\55.jpg"

img = Image.open(pic_path)  # 预测图片的路径
width, height = img.size[0], img.size[1]  # 保存图像的大小
img = transform(img)
img = torch.unsqueeze(img, dim=0)  # 扩展图像的维度

pred = net(img.to(device))  # 网络预测
pred = torch.squeeze(pred)  # 将(batch、channel)维度去掉
pred = np.array(pred.data.cpu())  # 保存图片需要转为cpu处理

pred[pred >= 0] = 255  # 处理结果二值化
pred[pred < 0] = 0

pred = np.uint8(pred)  # 转为图片的形式
pred = cv2.resize(pred, (width, height), cv2.INTER_CUBIC)  # 还原图像的size
cv2.imshow('pred', pred)
cv2.waitKey(0)
