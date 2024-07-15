
import torch
import torch.optim as optim
# from models import LSTM_SeqLabel,LSTM_SeqLabel_True
import argparse
from Modules.models import CTCVAE
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from baseline import saver
import time
import sys
import datetime
import Utils.utils as utils
import json


parser = argparse.ArgumentParser()

parser.add_argument('--status', default='train', choices=['train','generate'])
parser.add_argument('--msg', default='_')
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument("--exp_name", default='output' , type=str)
parser.add_argument("--load_ckt", default='output', type=str, choices=['none', 'output'])   #pre-train model  output
parser.add_argument("--load_ckt_loss", default="high", type=str, choices=['60', 'high', '70'])     #pre-train model
parser.add_argument("--path_train_data", default='emopia', type=str, choices=['emopia','ailabs'])
parser.add_argument("--data_root", default='./co-representation/', type=str)
parser.add_argument("--load_dict", default="dictionary.pkl", type=str)
parser.add_argument("--init_lr", default= 0.00001, type=float)
# inference config
parser.add_argument("--num_songs", default=1, type=int)
parser.add_argument("--emo_tag", default=2, type=int)
parser.add_argument("--out_dir", default='none', type=str)

args = parser.parse_args()


# 状态，训练还是测试
MODE = args.status
# hyper params
max_grad_norm = 3
n_epoch = args.epoch  # 默认100
batch_size = args.batch_size  # 默认4

###--- data ---###
path_data_root = args.data_root

# 数据集存储地址
path_train_data = os.path.join(path_data_root, args.path_train_data + '_data.npz')
path_dictionary =  os.path.join(path_data_root, args.load_dict)
path_train_idx = os.path.join(path_data_root, args.path_train_data + '_fn2idx_map.json')
path_train_data_cls_idx = os.path.join(path_data_root, args.path_train_data + '_idx.npz')

assert os.path.exists(path_train_data)
assert os.path.exists(path_dictionary)
assert os.path.exists(path_train_idx)


###--- training config ---###
# 模型存储地址
if MODE == 'train':
    path_exp = 'exp/' + args.exp_name

# 学习率
init_lr = args.init_lr  # 0.0001

# 预测是是否加载训练好模型
###--- fine-tuning & inference config ---###
if args.load_ckt == 'none':
    info_load_model = None
    print('NO pre-trained model used')
else:
    info_load_model = (
        # path to ckpt for loading
        'exp/' + args.load_ckt,
        # loss
        args.load_ckt_loss
        )

# 生成音乐存储地址
if args.out_dir == 'none':
    path_gendir = os.path.join('exp/' + args.load_ckt, 'gen_midis', 'loss_'+ args.load_ckt_loss)
else:
    path_gendir = args.out_dir


num_songs = args.num_songs
emotion_tag = args.emo_tag


class PEmoDataset(Dataset):
    def __init__(self):

        self.train_data = np.load(path_train_data)
        self.train_x = self.train_data['x']  # shape(1052, 1024, 8)  1052 首歌, 1024 个时间步, 8 个特征
        self.train_y = self.train_data[
            'y']  # shape(1052, 1024, 8)  1052 首歌, 1024 个时间步, 8 个特征   train_y[x] = train_x[x+1]  是下一个时间步预测
        self.train_mask = self.train_data['mask']

        # 标签转换  区分四象限  1-4
        self.cls_idx = np.load(path_train_data_cls_idx)  # 四个类别的索引
        self.cls_1_idx = self.cls_idx['cls_1_idx']
        self.cls_2_idx = self.cls_idx['cls_2_idx']
        self.cls_3_idx = self.cls_idx['cls_3_idx']
        self.cls_4_idx = self.cls_idx['cls_4_idx']


        self.train_x = torch.from_numpy(self.train_x).long()  # np转tensor
        self.train_y = torch.from_numpy(self.train_y).long()
        self.train_mask = torch.from_numpy(self.train_mask).float()

        self.seq_len = self.train_x.shape[1]  # 序列长度 1024
        self.dim = self.train_x.shape[2]  # 维度 8

        print('train_x: ', self.train_x.shape)

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index], self.train_mask[index]

    def __len__(self):
        return len(self.train_x)


# 准备torch的dataloader
def prep_dataloader(batch_size, n_jobs=0):
    dataset = PEmoDataset()

    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=False, drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader

def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

# 独热向量 可以把情绪改为一个向量
def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)

def train():
    myseed = 42069
    np.random.seed(myseed)
    torch.manual_seed(myseed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    # 加载数据集
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    train_loader = prep_dataloader(batch_size)


    # create saver  保存模型，loss等
    saver_agent = saver.Saver(path_exp)

    # config
    n_class = []  # number of classes of each token. [56, 127, 18, 4, 85, 18, 41, 5]  with key: [... , 25]
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))


    # 加载模型
    tcv_model = CTCVAE(n_class, True)

    # 训练
    tcv_model.cuda()
    tcv_model.train()
    n_parameters = network_paras(tcv_model)
    print('n_parameters: {:,}'.format(n_parameters))
    # 存储参数
    saver_agent.add_summary_msg(' > params amount: {:,d}'.format(n_parameters))

    # 是否加载模型中途训练  暂略

    # optimizers
    optimizer = optim.Adam(tcv_model.parameters(), lr=init_lr)

    # 跑训练
    n_token = len(n_class)
    start_time = time.time()
    for epoch in range(n_epoch):
        acc_loss = 0
        acc_losses = np.zeros(n_token)

        num_batch = len(train_loader)
        print('    num_batch:  分成了 {} 批:'.format(num_batch))


        for bidx, (batch_x, batch_y, batch_mask) in enumerate(train_loader):  # num_batch
            saver_agent.global_step_increment()

            # print(batch_x[0, 0:50, :], batch_y[0, 0:50, :])

            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_mask = batch_mask.cuda()

            # 改成VAE后，这儿返回recon_x, mu, sigma  => 用KS_Loss来算
            losses = tcv_model(batch_x, batch_y, batch_mask)

            # 类型type  emotion
            # VAE输入是 train_x, label，这个输入是train_x, train_y ；  这个应该怎么放入到VAE呢？  label直接是condition？  train_y中的emotion是一个变量

            # 怎么把trans和VAE融合呢？？

            loss = (losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5] + losses[6] + losses[7]) / 8

            # Update
            tcv_model.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write(
                '{}/{} | Loss: {:06f} | {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
                    bidx, num_batch, loss, losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], losses[6],
                    losses[7]))
            sys.stdout.flush()

            # acc
            acc_losses += np.array([l.item() for l in losses])
            acc_loss += loss.item()
            # log
            saver_agent.add_summary('batch loss', loss.item())

        # epoch loss
        runtime = time.time() - start_time
        epoch_loss = acc_loss / num_batch
        acc_losses = acc_losses / num_batch
        print('------------------------------------')
        print('epoch: {}/{} | Loss: {} | time: {}'.format(
            epoch, n_epoch, epoch_loss, str(datetime.timedelta(seconds=runtime))))

        each_loss_str = '{:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
            acc_losses[0], acc_losses[1], acc_losses[2], acc_losses[3], acc_losses[4], acc_losses[5], acc_losses[6],
            acc_losses[7])

        print('    >', each_loss_str)

        saver_agent.add_summary('epoch loss', epoch_loss)
        saver_agent.add_summary('epoch each loss', each_loss_str)

        # save model, with policy  保存模型
        loss = epoch_loss
        if 0.4 < loss <= 0.8:
            fn = int(loss * 10) * 10
            saver_agent.save_model(tcv_model, name='loss_' + str(fn))
        elif 0.08 < loss <= 0.40:
            fn = int(loss * 100)
            saver_agent.save_model(tcv_model, name='loss_' + str(fn))
        elif loss <= 0.08:
            print('Finished')
            return
        else:
            saver_agent.save_model(tcv_model, name='loss_high')


# 生成方法
def generate():
    # 加载模型pth
    # path
    path_ckpt = info_load_model[0]  # path to ckpt dir
    loss = info_load_model[1]  # loss
    name = 'loss_' + str(loss)
    path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # outdir 生成音乐存储地址
    os.makedirs(path_gendir, exist_ok=True)

    # config
    n_class = []  # num of classes for each token
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))
    n_token = len(n_class)

    # init model 初始化模型 用来生成
    ctv_model = CTCVAE(n_class, is_training=False)
    ctv_model.cuda()
    ctv_model.eval()
    print('[*] load model from:', path_saved_ckpt)

    ctv_model.load_state_dict(torch.load(path_saved_ckpt))

    n_parameters = network_paras(ctv_model)
    print('n_parameters: {:,}'.format(n_parameters))

    # 生成音乐
    start_time = time.time()
    song_time_list = []
    words_len_list = []

    cnt_tokens_all = 0
    sidx = 0
    while sidx < num_songs:
        # try:
        start_time = time.time()
        print('current idx:', sidx)

        if n_token == 8:
            path_outfile = os.path.join(path_gendir, 'emo_{}_{}'.format(str(emotion_tag), utils.get_random_string(10)))
            res, _ = ctv_model.generate_from_scratch(dictionary, emotion_tag, n_token)

        if res is None:
            continue
        np.save(path_outfile + '.npy', res)
        utils.write_midi(res, path_outfile + '.mid', word2event)

        song_time = time.time() - start_time
        word_len = len(res)
        print('song time:', song_time)
        print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)

        sidx += 1

    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))

    runtime_result = {
        'song_time': song_time_list,
        'words_len_list': words_len_list,
        'ave token time:': sum(words_len_list) / sum(song_time_list),
        'ave song time': float(np.mean(song_time_list)),
    }

    with open('runtime_stats.json', 'w') as f:
        json.dump(runtime_result, f)


if __name__ == '__main__':
    print("Hello, World!", args.status)

    if MODE == 'train':
        train()

    elif MODE == 'generate':
        generate()



