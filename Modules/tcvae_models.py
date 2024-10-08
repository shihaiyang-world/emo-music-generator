'''
1. 先把Transformer生成音符复现  baseline
2. 加入CVAE

'''

import torch.nn as nn
import math
import torch
import numpy as np
from fast_transformers.builders import TransformerEncoderBuilder as TransformerEncoderBuilder_local
from fast_transformers.builders import RecurrentEncoderBuilder as RecurrentEncoderBuilder_local
from fast_transformers.masking import TriangularCausalMask as TriangularCausalMask_local
import Utils.utils as utils
from torch.nn import functional as F

D_MODEL = 512
N_LAYER = 8
N_HEAD = 8


# 词嵌入
class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        # n_token 词典长度  d_model 词向量的向量长度
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 返回词向量 * 开根号d_model
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 计算每句话每个词的位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CTCVAE(nn.Module):

    def __init__(self, n_token, is_training=True, hidden_size=D_MODEL, output_size=64):
        super(CTCVAE, self).__init__()
        # --- params config --- #
        self.n_token = n_token  # == n_class  这是一个数组，记录每个特征的类别数 比如chord有5类n_token[chord]=5  这个会作为embedding第一个参数 词典长度
        # d_model 是指 隐向量长度 最后会全连接到这个维度
        self.d_model = D_MODEL
        self.n_layer = N_LAYER  #
        self.dropout = 0.1
        self.n_head = N_HEAD  #
        self.d_head = D_MODEL // N_HEAD
        self.d_inner = 2048
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        # 每个特征的embedding长度
        self.emb_sizes = [128, 256, 64, 32, 512, 128, 128, 128]

        # n_token是八个维度的词典长度 nclass = [56, 135, 18, 4, 87, 18, 42, 5]
        print('>>>>>:', self.n_token)
        # 8个特征的embedding 方法
        self.word_emb_tempo = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_type = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_pitch = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_duration = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.word_emb_velocity = Embeddings(self.n_token[6], self.emb_sizes[6])
        self.word_emb_emotion = Embeddings(self.n_token[7], self.emb_sizes[7])

        # 位置编码
        self.pos_emb = PositionalEncoding(self.d_model, self.dropout)

        # linear  最后特征全连接到这个d_model长度 隐向量
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)
        # blend with type
        self.project_concat_type = nn.Linear(self.d_model + 32, self.d_model)
        # encoder
        if is_training:
            # encoder (training)
            self.get_encoder('encoder')
        else:
            # encoder (inference)
            print(' [o] using RNN backend.')
            self.get_encoder('autoregred')

        # individual output
        self.proj_tempo = nn.Linear(self.d_model, self.n_token[0])
        self.proj_chord = nn.Linear(self.d_model, self.n_token[1])
        self.proj_barbeat = nn.Linear(self.d_model, self.n_token[2])
        self.proj_type = nn.Linear(self.d_model, self.n_token[3])
        self.proj_pitch = nn.Linear(self.d_model, self.n_token[4])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[5])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[6])
        self.proj_emotion = nn.Linear(self.d_model, self.n_token[7])


        self.output_size = output_size
        self.hidden_size = hidden_size
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)


    '''
    加入VAE部分
    '''
    # VAE 重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        # 随机一个
        eps = torch.randn_like(std)
        return mu + eps*std

    # VAE 带KL散度 loss
    def KL_loss_function(self, mu, logvar):
        # 判断是真或假
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def forward(self, x, target, loss_mask):

        # 经过embedding 和 encoder
        input_encoder, y_type = self.forward_hidden(x, is_training=True)





        # decoder 重构出来的结果，这个结果与真实目标target对比，产生loss
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion, emo_embd = self.forward_output(input_encoder, target)

        # reshape (b, s, f) -> (b, f, s)
        y_tempo = y_tempo[:, ...].permute(0, 2, 1)
        y_chord = y_chord[:, ...].permute(0, 2, 1)
        y_barbeat = y_barbeat[:, ...].permute(0, 2, 1)
        y_type = y_type[:, ...].permute(0, 2, 1)
        y_pitch = y_pitch[:, ...].permute(0, 2, 1)
        y_duration = y_duration[:, ...].permute(0, 2, 1)
        y_velocity = y_velocity[:, ...].permute(0, 2, 1)
        y_emotion = y_emotion[:, ...].permute(0, 2, 1)

        # loss
        loss_tempo = self.compute_loss(y_tempo, target[..., 0], loss_mask)
        loss_chord = self.compute_loss(y_chord, target[..., 1], loss_mask)
        loss_barbeat = self.compute_loss(y_barbeat, target[..., 2], loss_mask)
        loss_type = self.compute_loss(y_type, target[..., 3], loss_mask)
        loss_pitch = self.compute_loss(y_pitch, target[..., 4], loss_mask)
        loss_duration = self.compute_loss(y_duration, target[..., 5], loss_mask)
        loss_velocity = self.compute_loss(y_velocity, target[..., 6], loss_mask)
        loss_emotion = self.compute_loss(y_emotion, target[..., 7], loss_mask)

        # 返回各个Loss
        return loss_tempo, loss_chord, loss_barbeat, loss_type, loss_pitch, loss_duration, loss_velocity, loss_emotion, KL_loss_function



    # 计算出  隐向量
    def forward_hidden(self, x, memory=None, is_training=True):

        # embeddings
        emb_tempo =    self.word_emb_tempo(x[..., 0])
        emb_chord =    self.word_emb_chord(x[..., 1])
        emb_barbeat =  self.word_emb_barbeat(x[..., 2])
        emb_type =     self.word_emb_type(x[..., 3])
        emb_pitch =    self.word_emb_pitch(x[..., 4])
        emb_duration = self.word_emb_duration(x[..., 5])
        emb_velocity = self.word_emb_velocity(x[..., 6])

        emb_emotion = self.word_emb_emotion(x[..., 7])

        # 沿着最后一维堆叠
        embs = torch.cat(
            [
                emb_tempo,
                emb_chord,
                emb_barbeat,
                emb_type,
                emb_pitch,
                emb_duration,
                emb_velocity,
                emb_emotion

            ], dim=-1)

        # 全连接 把特征堆叠在一起后，跟n_model全连接
        # d_model 词向量的向量长度  所有embedding合在一起，全连接到这个长度了...
        emb_linear = self.in_linear(embs)
        # 加上了位置编码
        pos_emb = self.pos_emb(emb_linear)

        # transformer
        if is_training:
            # mask 邻接矩阵  句子中词是否mask  对角线矩阵
            attn_mask = TriangularCausalMask_local(pos_emb.size(1), device=x.device)
            # 定义Transformer结构
            embedding_encoder = self.transformer_encoder(pos_emb, attn_mask)  # y: b x s x d_model

            # project type  ？？？这是啥意思啊   是类型，是Node还是Mechanical
            y_type = self.proj_type(embedding_encoder)

            return embedding_encoder, y_type

        else:
            # 生成时，不需要mask
            pos_emb = pos_emb.squeeze(0)

            # self.get_encoder('autoregred')
            # self.transformer_encoder.cuda()
            embedding_encoder, memory = self.transformer_encoder(pos_emb, memory=memory)  # y: s x d_model

            # project type
            y_type = self.proj_type(embedding_encoder)
            return embedding_encoder, y_type, memory

    def forward_output(self, input_encoder, y_type):
        '''
        for training
        '''
        # tf_skip_emption = self.word_emb_emotion(y[..., 7])
        tf_skip_type = self.word_emb_type(y_type[..., 3])

        emo_embd = input_encoder[:, 0]

        # project other  沿着最后一维堆叠
        encoder_cat_type = torch.cat([input_encoder, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(encoder_cat_type)

        # individual output  做了一个全连接
        y_tempo = self.proj_tempo(y_)
        y_chord = self.proj_chord(y_)
        y_barbeat = self.proj_barbeat(y_)
        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)
        y_emotion = self.proj_emotion(y_)

        return y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion, emo_embd

    def get_encoder(self, TYPE):
        if TYPE == 'encoder':
            self.transformer_encoder = TransformerEncoderBuilder_local.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model // self.n_head,
                value_dimensions=self.d_model // self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()
        elif TYPE == 'autoregred':
            self.transformer_encoder = RecurrentEncoderBuilder_local.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model // self.n_head,
                value_dimensions=self.d_model // self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()

    # 生成
    def generate_from_scratch(self, dictionary, emotion_tag, key_tag=None, n_token=8, display=True):
        event2word, word2event = dictionary

        classes = word2event.keys()

        def print_word_cp(cp):

            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]

            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')

        generated_key = None

        target_emotion = [0, 0, 0, 1, 0, 0, 0, emotion_tag]

        init = np.array([
            target_emotion,  # emotion
            [0, 0, 1, 2, 0, 0, 0, 0]  # bar
        ])

        cnt_token = len(init)
        with torch.no_grad():
            final_res = []
            memory = None
            h = None

            cnt_bar = 1
            init_t = torch.from_numpy(init).long().cuda()
            print('------ initiate ------')

            for step in range(init.shape[0]):
                print_word_cp(init[step, :])
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])

                h, y_type, memory = self.forward_hidden(input_, memory, is_training=False)

            print('------ generate ------')
            while (True):
                # sample others
                next_arr, y_emotion = self.froward_output_sampling(h, y_type)
                if next_arr is None:
                    return None, None

                final_res.append(next_arr[None, ...])

                if display:
                    print('bar:', cnt_bar, end='  ==')
                    print_word_cp(next_arr)

                # forward
                input_ = torch.from_numpy(next_arr).long().cuda()
                input_ = input_.unsqueeze(0).unsqueeze(0)
                h, y_type, memory = self.forward_hidden(
                    input_, memory, is_training=False)

                # end of sequence
                if word2event['type'][next_arr[3]] == 'EOS':
                    break

                if word2event['bar-beat'][next_arr[2]] == 'Bar':
                    cnt_bar += 1

        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)

        return final_res, generated_key

    def froward_output_sampling(self, h, y_type, is_training=False):
        '''
        for inference
        '''

        # sample type
        y_type_logit = y_type[0, :]  # token class size
        cur_word_type = utils.sampling(y_type_logit, p=0.90, is_training=is_training)  # int
        if cur_word_type is None:
            return None, None

        if is_training:
            type_word_t = cur_word_type.long().unsqueeze(0).unsqueeze(0)
        else:
            # 生成
            type_word_t = torch.from_numpy(
                np.array([cur_word_type])).long().cuda().unsqueeze(0)  # shape = (1,1)

        tf_skip_type = self.word_emb_type(type_word_t).squeeze(0)  # shape = (1, embd_size)

        # concat
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        # project other
        y_tempo = self.proj_tempo(y_)
        y_chord = self.proj_chord(y_)
        y_barbeat = self.proj_barbeat(y_)

        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)
        y_emotion = self.proj_emotion(y_)

        # sampling gen_cond
        cur_word_tempo = utils.sampling(y_tempo, t=1.2, p=0.9, is_training=is_training)
        cur_word_barbeat = utils.sampling(y_barbeat, t=1.2, is_training=is_training)
        cur_word_chord = utils.sampling(y_chord, p=0.99, is_training=is_training)
        cur_word_pitch = utils.sampling(y_pitch, p=0.9, is_training=is_training)
        cur_word_duration = utils.sampling(y_duration, t=2, p=0.9, is_training=is_training)
        cur_word_velocity = utils.sampling(y_velocity, t=5, is_training=is_training)

        curs = [
            cur_word_tempo,
            cur_word_chord,
            cur_word_barbeat,
            cur_word_pitch,
            cur_word_duration,
            cur_word_velocity
        ]

        if None in curs:
            return None, None

        if is_training:
            cur_word_emotion = torch.from_numpy(np.array([0])).long().cuda().squeeze(0)
            # collect
            next_arr = torch.tensor([
                cur_word_tempo,
                cur_word_chord,
                cur_word_barbeat,
                cur_word_type,
                cur_word_pitch,
                cur_word_duration,
                cur_word_velocity,
                cur_word_emotion
            ])

        else:
            cur_word_emotion = 0

            # collect
            next_arr = np.array([
                cur_word_tempo,
                cur_word_chord,
                cur_word_barbeat,
                cur_word_type,
                cur_word_pitch,
                cur_word_duration,
                cur_word_velocity,
                cur_word_emotion
            ])

        return next_arr, y_emotion




