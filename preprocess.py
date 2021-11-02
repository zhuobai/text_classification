import torch
import numpy as np
import os
import pandas as pd
from collections import Counter
from gensim.models import KeyedVectors as Vectors
import json
import re

# 加载wiki数据集
def load_sst(dataset, filter = False):
    """
    dataset: 数据集名称
    filter: 是否清洗数据集，默认False
    """
    # 数据集文件地址
    data_dir = "data/" + dataset
    datas = []
    for data_name in ['train.tsv', 'dev.tsv', 'test.tsv']:
        data_file = os.path.join(data_dir, data_name)
        data = pd.read_csv(data_file, header=None, sep="\t", names=["sentence", "flag"], quoting=3)
        datas.append(data)

    return tuple(datas)


def cut(sentence):
    """
    将句子转化为小写，同时过滤标点符号等
    sentence:一个句子
    return: token list
    """
    # 去除标点符号
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
    sentence = re.sub(r,' ', sentence.lower()).split(' ')
    return [token for token in sentence if token != '']

def prepare(dataset, args):
    """
    dataset: 数据集
    args:config参数
    function: 保存词表和预训练词向量
    """
    # 创建词表
    word_freq = Counter()   # 单词词频
    for sentence in dataset["sentence"]:
        tokens = cut(sentence)      # 将单个句子分成单词
        word_freq.update(tokens)    # 更新词库：{"单词":数量}
    # 去掉词频小于规定的最小词频的单词
    words = [w for w in word_freq.keys() if word_freq[w] > args.min_word_freq]
    # 单词：词序
    word_map = {k:v+1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # 加载并保存预训练词向量
    pretrain_embed, embed_dim = load_embeddings(args.emb_file, word_map)
    embed = dict()
    embed['pretrain'] = pretrain_embed
    embed['dim'] = embed_dim
    torch.save(embed, os.path.join(args.output_folder, args.data + '_' + 'pretrain_embed.pth'))

    # 保存word_map
    with open(os.path.join(args.output_folder, args.data + '_' + 'wordmap.json'), 'w') as j:
        json.dump(word_map, j)


def init_embeddings(embeddings):
    """
    使用均匀分布U(-bias, bias)来随机初始化
    embeddings: 词向量矩阵
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)

def load_embeddings(emb_file, word_map):
    """
    emb_file: 预训练词向量文件路径
    word_map:词表
    return: 词向量矩阵， 词向量维度
    """
    # 单词表
    vocab = set(word_map.keys())
    print("Loading embedding...")
    cnt = 0  # 记录读入的词数

    # 使用gensim读取二进制词向量
    # vectors = Vectors.load_word2vec_format(emb_file, binary=True)
    # print("Load successfully")
    # emb_dim = 50
    # embeddings = torch.FloatTensor(len(vocab), emb_dim)
    # # 初始化词向量（对OOV进行随机初始化， 即对那些在词表上的词但不在预训练词向量中的词）
    # init_embeddings(embeddings)
    #
    # for emb_word in vocab:
    #     if emb_word in vectors.index_to_key:
    #         # 单词在预训练词表中
    #         embedding = vectors[emb_word]   # 对应单词的预训练词向量
    #         cnt += 1
    #         embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)
    #
    #     else:
    #         continue
    #
    # print("Number of words read:", cnt)
    # print("Number of OOV:", len(vocab)-cnt)
    #
    # return embeddings, emb_dim

    # 读取glove词向量
    with open(emb_file, 'r', encoding='utf-8') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    # 初始化词向量(对OOV进行随机初始化，即对那些在词表上的词但不在预训练词向量中的词)
    init_embeddings(embeddings)

    # 读入词向量文件
    for line in open(emb_file, 'r', encoding='utf-8'):
        line = line.split(' ')
        emb_word = line[0]

        # 过滤空值并转为float型
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # 如果不在词表上
        if emb_word not in vocab:
            continue
        else:
            cnt += 1

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    print("Number of words read: ", cnt)
    print("Number of OOV: ", len(vocab) - cnt)

    return embeddings, emb_dim

def tokens_to_idx(sentence, word_map, args):
    """
    将token转为索引
    sentence:句子
    word_map:词表
    args:config参数
    type:句子类型（问题或答案）
    return: index list
    """
    max_len = args.q_max_len

    # 将句子分割成单词列表
    tokens = cut(sentence)
    return [word_map.get(word, word_map['<unk>']) for word in tokens] + [word_map['<pad>']] * (max_len - len(tokens))

def position_index(sentence, length):
    """
    获得句子位置索引
    :param sentence:句子
    :param length: 句子长度
    :return: 句子位置索引
    """
    index = np.zeros(length)

    raw_len = len(cut(sentence))
    index[:min(raw_len, length)] = range(1, min(raw_len + 1, length + 1))
    # print index
    return index

def load_data(dataset, word_map, args):
    """
    将单词序列转成数字序列的数据集
    dataset: 原始数据集
    word_map: 词表
    args:config参数
    """
    # 将问题token转成索引
    dataset["token_idx"] = dataset["sentence"].apply(lambda x: tokens_to_idx(x, word_map, args))
    # 位置索引
    dataset["q_position"] = dataset["sentence"].apply(lambda x: position_index(x, args.q_max_len))

    q = torch.LongTensor(dataset["token_idx"])
    l = torch.LongTensor(dataset["flag"])
    qp = torch.LongTensor(dataset["q_position"])
    return q, l, qp




