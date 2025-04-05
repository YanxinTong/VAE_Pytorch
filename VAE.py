# 该模块主要是为了实现VAE模型的，

'''
# Part1 引入相关的库函数
'''
import torch
from torch import nn
from dataset import Mnist_dataset

'''
# Part2 设计AE的类函数
'''

class VAE(nn.Module):
    def __init__(self,img_channel,img_size,encode_f1_size,latent_size):
        super().__init__()

        self.encode=nn.Sequential(
            nn.Linear(in_features=img_channel*img_size**2,out_features=encode_f1_size),
            nn.ReLU(),
            nn.Linear(in_features=encode_f1_size,out_features=latent_size*2),
            nn.ReLU()
        )

        self.decode=nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=encode_f1_size),
            nn.ReLU(),
            nn.Linear(in_features=encode_f1_size, out_features=img_channel * img_size ** 2),
            nn.Sigmoid()
        )

    def forward(self,x):
        input1=x.view(x.size()[0],-1) # (batch,channel*img_size*img_size)
        mid_latent=self.encode(input1) # (batch,latent_size)
        mu,sigma=torch.chunk(mid_latent,chunks=2,dim=1)

        # 重参数化
        latent=mu+sigma*torch.randn_like(sigma) # 利用均值和标准差，生成对应的latent取样值
        result=self.decode(latent) # (batch,channel*img_size*img_size)
        result=result.view(*(x.size()))
        return result,mu,sigma

'''
# 开始测试
'''
if __name__ == '__main__':
    img,label=Mnist_dataset[0]
    vae=VAE(img_channel=1,img_size=28,encode_f1_size=400,latent_size=10)
    result=vae(img)
    print(result.size())