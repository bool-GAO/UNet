# train.py对定义好的网络进行训练
from torch import nn,optim
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path='./unet.pth' # 训练好的网络模型参数保存地址
data_path=r'data'
save_path='train_image' #保存训练好的图片
if __name__ == '__main__':
    data_loader=DataLoader(MyDataset(data_path),batch_size=4,shuffle=True)
    net=UNet().to(device)
    # 判断之前是否进行过训练，选择是否加载训练权重文件
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

	# 定义优化和损失函数
    opt=optim.Adam(net.parameters())
    loss_fun=nn.BCELoss()
	# 进行训练
    epoch=1
    while True:
        for i,(image,segment_image) in enumerate(data_loader):
            image, segment_image=image.to(device),segment_image.to(device)

            out_image=net(image)
            train_loss=loss_fun(out_image,segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i%5==0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            if i%50==0:
                torch.save(net.state_dict(),weight_path)

            _image=image[0]
            _segment_image=segment_image[0]
            _out_image=out_image[0]

            img=torch.stack([_image,_segment_image,_out_image],dim=0)
            save_image(img,f'{save_path}/{i}.png')

        epoch+=1


