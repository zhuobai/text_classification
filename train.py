import torch
import torch.nn as nn
import config
from preprocess import load_sst, prepare, load_data
from datasets import MyDataset
from model.cnn import CNNModel, LSTMModel
import torch.utils.data as Data
import os
import time
import json
from tqdm import tqdm
from torch.functional import F


def test_point_wise():
    train_data, dev, test = load_sst(args.data)
    print("数据量：",len(train_data))

    # 问题句子最大长度
    q_max_sent_length = max(map(lambda x:len(x), train_data['sentence'].str.split()))
    args.q_max_len = q_max_sent_length
    print("question_max_length: ", args.q_max_len)

    print('train length:', len(train_data))
    print('test length:', len(test))
    print('dev length', len(dev))

    # 词表文件位置
    word_map_file = os.path.join(args.output_folder, args.data + '_' + 'wordmap.json')
    # 预训练词向量文件
    embed_file  = os.path.join(args.output_folder, args.data + '_' + 'pretrain_embed.pth')
    # 判断词表文件和预训练词向量文件是否存在（注：这俩文件是同时生成的）
    if not os.path.isfile(word_map_file):
        # 生成词表文件和预训练词向量文件
        prepare(train_data, args)
    # 读入词表
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    
    vocab = set(word_map.keys())
    vocab_size = len(vocab)
    args.vocab_size = vocab_size
    # 加载预训练词向量
    embed_file = torch.load(embed_file)
    pretrain_embed, args.embed_dim = embed_file['pretrain'], embed_file['dim']

    # 生成数字形式的数据集
    q, l, qp = load_data(train_data, word_map, args)
    train_datasets = MyDataset(q, l, qp)
    # dataloader:封装数据集
    train_loader = Data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)
    
    # 模型
    if args.model == "CNNModel":
        model = CNNModel(args, pretrain_embed).to(args.device)
    else:
        model = LSTMModel(args, pretrain_embed).to(args.device)
    # 训练和验证
    train(train_loader, dev, test, model, word_map)

# 打印模型参数
def print_model(model):
    cnt = 0
    for name, param in model.named_parameters():
        print(name, param)
        
        
        
def train(train_loader, dev, test, model, word_map):
    # 随机梯度
    opt = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.l2_regularization)
    criterion = nn.CrossEntropyLoss()
    max_accuracy = 0
    batch_nums = len(train_loader)
    max_map = 0.67
    with open(precision, "w") as log:
        # log.write(str(vars(args).items()) + '\n')
        for epoch in range(args.epochs):
            model.train()
            total_loss, total_samples = 0.0, 0.0
            pre_nums = 0.0 # 预测正确样例数
            for i,batch in enumerate(tqdm(train_loader)):
                q,  labels, q_position = [i.to(args.device) for i in batch]

                opt.zero_grad()
                # 经过模型返回的分类结果（batch_size, 2)
                logits = model(q, q_position)
                loss = criterion(logits, labels)
                total_loss += loss
                # 反向传播
                loss.backward()
                opt.step()
                scores = F.softmax(logits,dim=-1)

                # 得到预测的结果
                predictions =torch.argmax(scores, dim=-1)
                pre_nums += torch.sum(torch.eq(predictions, labels).float())
                total_samples += len(q)

            acc = pre_nums / total_samples
            loss_mean = total_loss / batch_nums
            print(f'epoch: {epoch}')
            print(f'train   loss: {loss_mean:.4f}   |  acc: {acc * 100:.2f}')

            # 验证集
            # map_mrr_dev = eval(dev, model, criterion, word_map)
            # print("{}:dev epoch:map mrr {}".format(epoch, map_mrr_dev))
            #
            # 测试集
            eval(test, model, criterion, word_map)
            #
            # line = " {}:epoch: map_dev{}".format(epoch, map_mrr_test)
            # log.write(line + '\n')
            # log.flush()
            # if map_mrr_dev[0] > max_map:
            #     torch.save(model,args.save_model)
        # log.close()
# 验证集
def eval(dev, model, criterion, word_map):
    # 生成数字列表的数据集
    q, l, qp = load_data(dev, word_map, args)
    dev_datasets = MyDataset(q, l, qp)
    # 封装数据集
    dev_loader = Data.DataLoader(dev_datasets,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    model.eval()
    total_loss = 0
    batch_num = len(dev_loader)  # batch的数量
    acc_num = 0 # 预测正确的样本数量
    total_data = 0 # 全部数据
    scores = []
    with torch.no_grad():  # 不需要更新模型，不需要梯度
        for i, batch in enumerate(tqdm(dev_loader,leave=False)):
            q, labels, q_position= [i.to(args.device) for i in batch]
            logits = model(q, q_position)
            example_losses = criterion(logits, labels)

            # softmax归一化
            score = F.softmax(logits, dim=-1)
            preds = torch.argmax(score, dim=-1)
            scores.extend(score.cpu().data.numpy())
            acc_num += torch.sum(preds == labels)
            total_loss += example_losses.item()
            total_data += len(q)

    acc = acc_num.item() / total_data
    loss = total_loss / batch_num
    print(f'test     loss: {loss:.4f}   |  acc: {acc*100:.2f}\n')

def set_seed(seed):
    torch.manual_seed(seed) # 为cpu设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子


if __name__ == '__main__':
    # 获取参数
    args = config.get_args()
    # 设置随机种子
    set_seed(args.seed)
    now = int(time.time())
    timeArray = time.localtime(now)
    timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
    timeDay = time.strftime("%Y%m%d", timeArray)
    # 日志文件位置
    log_dir = args.data + '_log/' + timeDay
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    para_file = log_dir + '/test_'+args.data + timeStamp + '.txt'
    data_file = log_dir + '/test_'+args.data + timeStamp
    precision = data_file + 'precise'

    test_point_wise()