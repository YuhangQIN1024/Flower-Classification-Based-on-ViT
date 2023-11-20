import torch
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


import os
from PIL import Image
#path_img = 'C:\\Users\\12646\\PycharmProjects\\VIT_16\\plotimg'
#img_dir = os.listdir(path_img)
#print(img_dir)
#print(len(img_dir))
#for i in range(len(img_dir)):

#    id = img_dir[i].split('.')[0]
 #   img = Image.open(path_img + '/' + img_dir[i])
  #  size_img = img.size
   # print(size_img)
    #weight = int(size_img[0] // 3)
    #height = int(size_img[1] // 3)
    #for j in range(3):
    #    for k in range(3):
    #        box = (weight * k, height * j, weight * (k + 1), height * (j + 1))
    #        region = img.crop(box)
    #        region.save('C:\\Users\\12646\\Desktop\\1''{}-{}{}.png'.format(id, j, k))
img = Image.open("./plotimg/3.jpg")
print("原图大小：",img.size)
data1 = transforms.RandomResizedCrop(224)(img)
print("随机裁剪后的大小:",data1.size)
data2 = transforms.RandomResizedCrop(224)(img)
data3 = transforms.RandomResizedCrop(224)(img)

plt.subplot(2,2,1),plt.imshow(img),plt.title("原图")
plt.subplot(2,2,2),plt.imshow(data1),plt.title("转换后的图1")
plt.subplot(2,2,3),plt.imshow(data2),plt.title("转换后的图2")
plt.subplot(2,2,4),plt.imshow(data3),plt.title("转换后的图3")
plt.show()
