# utils.py文件保存一些工具函数
from PIL import Image
def keep_image_size_open(path, size=(256, 256)):
    #处理图片的大小
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))  # 'RGB'
    mask.paste(img, (0, 0)) #将图片粘贴到左上角
    mask = mask.resize(size)
    return mask
