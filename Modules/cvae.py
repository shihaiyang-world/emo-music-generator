import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# cuda setup
device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': True}

# hyper params
batch_size = 64
latent_size = 20
epochs = 10


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False, **kwargs)


def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        # 全连接input size: feature_size + class_size (condition)  output size: 400
        self.fc1  = nn.Linear(feature_size + class_size, 400)
        # 全连接input size: 400  output size: latent_size
        self.fc21 = nn.Linear(400, latent_size)
        # 全连接input size: 400  output size: latent_size
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        # 全连接input size: latent_size + class_size (condition)  output size: 400
        self.fc3 = nn.Linear(latent_size + class_size, 400)
        # 全连接input size: 400  output size: feature_size  还原回原尺寸
        self.fc4 = nn.Linear(400, feature_size)

        # 与 Leaky-ReLU 和 PReLU 类似，与 ReLU 不同的是，ELU 没有神经元死亡的问题(ReLU Dying 问题是指当出现异常输入时，在反向传播中会产生大的梯度，这种大的梯度会导致神经元死亡和梯度消失)。 它已被证明优于 ReLU 及其变体，如 Leaky-ReLU(LReLU) 和 Parameterized-ReLU(PReLU)。 与 ReLU 及其变体相比，使用 ELU 可在神经网络中缩短训练时间并提高准确度。
        # 两个激活函数
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c) c就是condition
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        # 将x和c拼接在一起  特征x和condition c合并
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        # 全连接 激活  output size: 400
        h1 = self.elu(self.fc1(inputs))
        # 全连接 激活  output size: latent_size    算均值mu
        z_mu = self.fc21(h1)
        # 全连接 激活  output size: latent_size  # 算方差sigma
        z_var = self.fc22(h1)
        return z_mu, z_var

    # 重参数化技巧
    # 实现都一样
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        # 随机一个
        eps = torch.randn_like(std)
        return mu + eps*std

    # 通过降为后的隐向量，去生成图片
    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        # 隐向量和condition合并
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        # 全连接 激活  output size: 400
        h3 = self.elu(self.fc3(inputs))
        # 全连接 激活  output size: feature_size  还原回原尺寸
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 28*28), c)
        # z就是从高斯分布采样的值，用这个值来decoder还原。
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# create a CVAE model
model = CVAE(28*28, latent_size, 10).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # 判断是真或假
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        labels = one_hot(labels, 10)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            labels = one_hot(labels, 10)
            recon_batch, mu, logvar = model(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
            if i == 0:
                n = min(data.size(0), 5)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'reconstruction_' + str(f"{epoch:02}") + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            c = torch.eye(10, 10).cuda()
            sample = torch.randn(10, latent_size).to(device)
            sample = model.decode(sample, c).cpu()
            save_image(sample.view(10, 1, 28, 28),
                       'sample_' + str(f"{epoch:02}") + '.png')



