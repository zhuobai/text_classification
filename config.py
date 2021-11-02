import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    # type 是要传入的参数的数据类型， help是该参数的提示信息
    parser.add_argument('--batch_size', type=int, default=60, help="批处理数量")
    parser.add_argument('--epochs', type=int, default=20, help="训练次数")
    parser.add_argument('--embed', type=int, default=300, help="词向量大小")
    parser.add_argument('--lr', type=float, default=0.01, help="学习率")
    parser.add_argument('--dropout', type=float, default=0.3, help="dropout")
    parser.add_argument('--hidden_size', type=int, default=100, help="隐藏层大小")
    parser.add_argument('--model', type=str, default="CNNModel", help="模型类型")
    parser.add_argument('--data', type=str, default="sst2", help="数据集")
    parser.add_argument('--l2_regularization', type=float, default=0.0001, help="正则化参数")
    parser.add_argument('--q_max_len', type=int, default=200, help="问题句子长度")
    parser.add_argument('--a_max_len', type=int, default=200, help="答案句子长度")
    parser.add_argument("--emb_file", type=str, default='data/glove.840B.300d.txt')
    parser.add_argument('--output_folder', type=str, default="output_data", help="处理后的数据保存的文件路径")
    parser.add_argument('--min_word_freq', type=int, default=3, help="最小词频数")
    parser.add_argument('--kernel_num', type=int, default=100, help="卷积核数量")
    parser.add_argument('--kernel_sizes', type=list, default=[3,4,5], help="不同尺寸的kernel")
    parser.add_argument('--vocab_size', type=int, default=1000, help="单词表数量")
    parser.add_argument('--seed', type=int, default=123, help="随机种子")
    parser.add_argument('--device', help="使用gpu")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return args

