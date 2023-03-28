import time

import cv2
import time
from net import *
from data import *

if __name__ == '__main__':
    net=UNet().cuda()

    weights='./unet.pth'
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully')
    else:
        print('no loading')

    _input='./data/test/13.bmp'

    time_start = time.time()
    img=keep_image_size_open(_input) #三通道RGB
    img_data=transform(img).cuda()
    img_data=torch.unsqueeze(img_data,dim=0)
    net.eval()
    time_start = time.time()
    for i in range(100):
        out=net(img_data)
    time_end = time.time()
    out=torch.argmax(out,dim=1)
    out=torch.squeeze(out,dim=0)
    out=out.unsqueeze(dim=0)
    print(set((out).reshape(-1).tolist()))
    out=(out).permute((1,2,0)).cpu().detach().numpy()
    time_end = time.time()
    print(time_end-time_start,'s')
    cv2.imwrite('./result/result.png',out*255.0)
    cv2.imshow('out',out*255.0)
    cv2.waitKey(0)

