# 训练模型，该模块主要是为了实现对于模型的训练，
'''
# Part1 引入相关的库函数
'''

import torch
from torch import nn
from dataset import Mnist_dataset
from VAE import VAE
import torch.utils.data as data

'''
初始化一些训练参数
'''
EPOCH = 50
Mnist_dataloader = data.DataLoader(dataset=Mnist_dataset, batch_size=64, shuffle=True)

# 前向传播的模型
net = VAE(img_channel=1, img_size=28, encode_f1_size=400, latent_size=10)

# 计算损失函数,VAE和AE不同的点，还在于，需要计算正态分布之间的KL散度。

# 反向更新参数
lr = 1e-3
optim = torch.optim.Adam(params=net.parameters(), lr=lr)


# 定义VAE的损失函数，主要包含重建损失和KL散度
def vae_loss(rec_x, x, mu, sigma):
    # 首先是重建损失
    loss1 = nn.BCELoss(reduction='sum')
    rec_loss = loss1(rec_x, x)
    # 然后是KL散度的损失，也就是预测出来的均值和方差要满足标准正态分布(所以衡量的是标准正态分布和预测到的分布的差距和),这里是假设log以2为底
    KL_loss = -0.5 * torch.sum(1 + sigma - torch.pow(mu, 2) - torch.pow(sigma, 2))
    return rec_loss + KL_loss


'''
# 开始训练
'''
# net.train() # 设置为训练模式

for epoch in range(EPOCH):
    n_iter = 0
    for batch_img, _ in Mnist_dataloader:
        # 先进行前向传播
        batch_img_pre, mu, sigma = net(batch_img)  #

        # 计算损失
        loss_cal = vae_loss(batch_img_pre, batch_img, mu, sigma)

        # 清除梯度
        optim.zero_grad()
        # 反向传播
        loss_cal.backward()
        # 更新参数
        optim.step()

        l = loss_cal.item()

        if n_iter % 100 == 0:
            print('此时的epoch为{},iter为{},loss为{}'.format(epoch, n_iter, l))

        n_iter += 1
    if epoch == 20:
        # 注意pt文件是保存整个模型及其参数的，pth文件只是保存参数
        torch.save(net.encode, 'VAE_encoder_eopch_{}.pt'.format(epoch))
        # 注意pt文件是保存整个模型及其参数的，pth文件只是保存参数
        torch.save(net.decode, 'VAE_decoder_eopch_{}.pt'.format(epoch))
        break
