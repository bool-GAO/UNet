#  data.py文件用来进行数据集的制作
import os
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
transform=transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path=path
        self.name=os.listdir(os.path.join(path,'mask')) # 拼接取出SegmentationClass文件夹下面的所有图片

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        mask_name=self.name[index]  #mask图片的名称
        mask_path=os.path.join(self.path,'mask',mask_name)
        image_path=os.path.join(self.path,'train',mask_name.replace('bmp','bmp'))
        mask_image=keep_image_size_open(mask_path)
        image=keep_image_size_open(image_path) # utils包下面的函数，用来处理图片的大小（将大小不一的图片变成同样的像素值便于之后的训练）
        return transform(image),transform(mask_image) # 变成Tensor

if __name__ == '__main__':
    data=MyDataset('./data')
    print(data[0][0].shape)
    print(data[0][1].shape)
