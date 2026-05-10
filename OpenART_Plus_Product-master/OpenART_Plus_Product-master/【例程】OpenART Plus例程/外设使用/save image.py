from machine import UART
import pyb
import struct
from pyb import LED
import sensor, image, time, math, tf
import os


red = LED(1)
green = LED(2)
blue = LED(3)

sensor.reset()
sensor.set_pixformat(sensor.RGB565)     # 设置摄像头像素格式
sensor.set_framesize(sensor.QVGA)      # 设置摄像头分辨率
sensor.set_brightness(1000)             # 设置摄像头亮度 越大越亮
sensor.set_auto_whitebal(True)
sensor.skip_frames(time = 20)

clock = time.clock()

save_img_num = 0;
while(True):
    img = sensor.snapshot()             # 获取一幅图像
    blue.toggle()                       # 蓝灯翻转

    # 修改文件名称，准备保存
    save_img_num += 1;
    image_pat = "/sd/"+str(save_img_num)+".jpg"

    # 将拷贝之后的图像保存到sd卡
    img.save(image_pat,quality=99)


