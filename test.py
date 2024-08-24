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
# 读取所有图片路径
tests_path = os.listdir('./test/')  # 获取 './predict/' 路径下所有文件,这里的路径只是里面文件的路径
''''
print(tests_path)
['0.png', '1.png', '10.png', '11.png', '12.png', '13.png', '14.png', 
'15.png', '16.png', '17.png', '18.png', '19.png', '2.png', '20.png', 
'21.png', '22.png', '23.png', '24.png', '25.png', '26.png', '27.png',
 '28.png', '29.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png']
'''
print(tests_path)
with torch.no_grad():  # 预测的时候不需要计算梯度
    for test_path in tests_path:  # 遍历每个predict的文件
        save_pre_path = './result/' + test_path.split('.')[-2] + '_res.png'  # 将保存的路径按照原图像的后缀，按照数字排序保存
        img = Image.open('./test/' + test_path)  # 预测图片的路径
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
        cv2.imwrite(save_pre_path, pred)  # 保存图片


