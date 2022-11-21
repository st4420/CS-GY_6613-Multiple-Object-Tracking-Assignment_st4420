import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet
# -*- coding:utf-8 -*-
import ctypes

import predict

    # 选择设备，有cuda用cuda，没有就用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 加载网络，图片单通道，分类为1。
net = UNet(n_channels=1, n_classes=1)
# 将网络拷贝到deivce中
net.to(device=device)
# 加载模型参数
net.load_state_dict(torch.load('model.pth', map_location=device))
# 测试模式
net.eval()
def process(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    origin_shape = img.shape
    img = cv2.resize(img, (512, 512))
    # 转为batch为1，通道为1，大小为512*512的数组
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
# 转为tensor
    img_tensor = torch.from_numpy(img)
# 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
# 预测
    pred = net(img_tensor)
# 提取结果
    pred = np.array(pred.data.cpu()[0])[0]
# 处理结果
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
# 保存图片
    pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
    width = origin_shape[1]
    height = origin_shape[0]
    pred0=np.zeros((height, width), np.uint8)
    for i0 in range(0,origin_shape[0]):
        for i1 in range(0,origin_shape[1]):
            pred0[i0,i1]=pred[i0,i1]
    return pred0

def capture_video():
    capture = cv2.VideoCapture('1.mp4')
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        cv2.imshow('video', frame)
        pred=process(frame)

        cv2.imshow('pred', pred)

        if cv2.waitKey(20) == 27:
            break


capture_video()

cv2.waitKey(0)
cv2.destroyAllWindows()
