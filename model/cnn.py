import torch
from torch import nn
import torch.nn.functional as F

class CNNModel(nn.Module):

    def __init__(self, config, word_emb):
        super(CNNModel, self).__init__()
        self.config = config
        # 预训练词向量
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.embedding.weight.requires_grad_()
        # 初始化为单通道
        channel_num = 1

        # 卷积层, kernel size: (size, embed_dim), output: [(batch_size, kernel_num, h,1)]
        self.convs = nn.ModuleList([
            nn.Conv2d(channel_num, config.kernel_num, (size, config.embed))
            for size in config.kernel_sizes
        ])

        self.dropout = nn.Dropout(config.dropout)

        # 全连接层
        self.fc = nn.Linear(len(config.kernel_sizes) * config.kernel_num, 2)

        self.fc1 = nn.Linear(config.embed,20)
        self.fc2 = nn.Linear(20 * config.q_max_len, 2)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x, q_position):
        x = self.embedding(x).unsqueeze(1)  # (batch_size, 1, max_len, word_vec)
        # 卷积
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, kernel_num, h)]
        # 池化
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, kernel_num)]

        # flatten
        x = torch.cat(x, 1)  # (batch_size, kernel_num * len(kernel_sizes))
        # dropout
        x = self.dropout(x)
        # fc
        x = self.fc(x)  # logits, 没有softmax, (batch_size, class_num)
        return x


class Attn(nn.Module):
    '''
    Attention Layer
    '''

    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        '''
        :param x: (batch_size, max_len, hidden_size)
        :return alpha: (batch_size, max_len)
        '''
        x = torch.tanh(x)  # (batch_size, max_len, hidden_size)
        x = self.attn(x).squeeze(2)  # (batch_size, max_len)
        alpha = F.softmax(x, dim=1).unsqueeze(1)  # (batch_size, 1, max_len)
        return alpha


class LSTMModel(nn.Module):
    '''
    BiLSTM: BiLSTM, BiGRU
    '''

    def __init__(self, config, word_emb):

        super(LSTMModel, self).__init__()
        self.config = config
        # 预训练词向量
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.embedding.weight.requires_grad_()
        self.hidden_size = config.hidden_size

        self.dropout = nn.Dropout(config.dropout)

        self.bilstm = nn.LSTM(config.embed, config.hidden_size, dropout=config.dropout,
                                  bidirectional=True, batch_first=True)

        self.fc = nn.Linear(config.hidden_size, 2)

        self.attn = Attn(config.hidden_size)

    def forward(self, x, position):
        '''
        :param x: [batch_size, max_len]
        :return logits: logits
        '''
        x = self.embedding(x)  # (batch_size, max_len, word_vec)
        x = self.dropout(x)
        # 输入的x是所有time step的输入, 输出的y实际每个time step的hidden输出
        # _是最后一个time step的hidden输出
        # 因为双向,y的shape为(batch_size, max_len, hidden_size*num_directions), 其中[:,:,:hidden_size]是前向的结果,[:,:,hidden_size:]是后向的结果
        y, _ = self.bilstm(x)  # (batch_size, max_len, hidden_size*num_directions)
        y = y[:, :, :self.hidden_size] + y[:, :, self.hidden_size:]  # (batch_size, max_len, hidden_size)
        alpha = self.attn(y)  # (batch_size, 1, max_len)
        r = alpha.bmm(y).squeeze(1)  # (batch_size, hidden_size)
        h = torch.tanh(r)  # (batch_size, hidden_size)
        logits = self.fc(h)  # (batch_size, class_num)
        logits = self.dropout(logits)
        return logits


class CMPModel(nn.Module):

    def __init__(self, config, word_emb):
        super(QAModel, self).__init__()
        self.config = config
        # 预训练词向量
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.embedding.weight.requires_grad_()
        # 复数词向量
        self.cmp_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        # 概率参数
        self.weighted_q = nn.Parameter(torch.randn(1, config.q_max_len, 1, 1))
        self.weighted_a = nn.Parameter(torch.randn(1, config.a_max_len, 1, 1))
        self.weighted_q_att = nn.Parameter(torch.randn(1, config.q_max_len, 1, 1))
        self.weighted_a_att = nn.Parameter(torch.randn(1, config.a_max_len, 1, 1))

    def concat_embedding(self, words_indice, position_indice):
        # 实数部分词向量:[batch_size, seq_len, embedding_size]
        embedded_real = self.dropout(self.embedding(words_indice))
        # 虚数部分词向量;[batch_size, seq_len, embedding_size]
        embedded_complex = self.dropout(self.cmp_embedding(words_indice))
        pos = position_indice.unsqueeze(dim=-1)
        embedded_complex = torch.mul(embedded_complex, pos)

        return embedded_real, embedded_complex

    def density_matrix(self, sentence_matrix, sentence_matrix_complex, sentence_weighted):
        # 扩充维度:[batch_size, seq_len, dim, 1]
        input_real = sentence_matrix.unsqueeze(dim=-1)
        input_imag = sentence_matrix_complex.unsqueeze(dim=-1)
        # 转换矩阵 [batch_size, seq_len, 1, dim]
        input_real_transpose = input_real.transpose(3, 2)
        # 复数共轭转置，虚部要乘以-1
        input_imag_transpose = input_imag.transpose(3, 2) * (-1)
        # 求密度矩阵的实部
        real_real = torch.matmul(input_real, input_real_transpose)
        real_imag = torch.matmul(input_imag, input_imag_transpose)
        # 得到复数的实部，运算符号是减号
        real = real_real - real_imag
        # 求密度矩阵的虚部
        imag_real = torch.matmul(input_imag, input_real_transpose)
        imag_imag = torch.matmul(input_real, input_imag_transpose)
        # 得到复数的虚部，运算符号是加号
        imag = imag_real + imag_imag
        return torch.sum(torch.mul(real, sentence_weighted), 1), torch.sum(torch.mul(imag, sentence_weighted), 1)

    def forward(self, questions, q_position):
        # 获得实数词向量和复数词向量 [batch_size, seq_len, embedding_size]
        embedded_q_real, embedded_q_complex = self.concat_embedding(questions, q_position)
        # 密度矩阵
        density_q_real, density_q_imag = self.density_matrix(embedded_q_real, embedded_q_complex, self.weighted_q)









