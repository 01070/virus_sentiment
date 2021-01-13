import xlrd
import numpy as np
from data_utils import *
import logging
import os
import torch.nn as nn
import torch
import math
import argparse
import sys
import pandas
from sklearn.metrics import accuracy_score, f1_score
import random
import os
from time import strftime, localtime
import matplotlib.pylab as plt
import pandas as pd
from torch.cuda.amp import autocast as autocast
from Bert_weibo import *
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, AutoTokenizer, BertModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class SSTdataset(Dataset):
    def __init__(self, opt, dataset_class, tokenizer):
        if dataset_class == 'train':
            dataset_file = os.path.join(opt.dataset_file, 'train.tsv')
        else:
            dataset_file = os.path.join(opt.dataset_file, 'test.tsv')

        content = pd.read_csv(dataset_file, sep='\t')
        all_data = []
        for sentence, label in zip(content['sentence'], content['label']):
            # sentence_token = tokenizer.text_to_sequence(sentence) if 'Bert' not in opt.model_name else tokenizer.encode(sentence, max_length=opt.max_seq_len, padding='max_length', truncation=True)
            sentence_token = tokenizer.encode(sentence, max_length=opt.max_seq_len, padding='max_length', truncation=True)
            sentence_token = np.array(sentence_token)
            data = {
                'text_indices': sentence_token,
                'polarity': label
            }
            all_data.append(data)

            self.data = all_data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class UsualDataset(Dataset):
    def __init__(self, opt, dataset_class):
        if dataset_class == 'train':
            dataset_file = os.path.join(opt.dataset_file, 'train.xlsx')
        else:
            dataset_file = os.path.join(opt.dataset_file, 'test.xlsx')
        tokenizer = BertTokenizer.from_pretrained(opt.pretrained_model)
        wordbook = xlrd.open_workbook(dataset_file)
        sheet = wordbook.sheet_by_index(0)
        all_data = []
        content = sheet.col_values(1, 1)
        lable = sheet.col_values(2, 1)
        for x, y in zip(content, lable):
            text_indices = tokenizer.encode(x, max_length=200, padding='max_length', truncation=True)
            text_indices = np.array(text_indices)
            polarity = opt.polarity_dict[y]
            data = {
                'text_indices': text_indices,

                'polarity': polarity

            }
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if opt.dataset == 'sst2' and opt.model_name == 'TCN':
            tokenizer = build_tokenizer(
                fnames=[os.path.join(opt.dataset_file, 'train.tsv'),
                        os.path.join(opt.dataset_file, 'train.tsv')],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.input_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.input_dim), opt.dataset))
            self.train_data = SSTdataset(opt, 'train', tokenizer)
            self.test_data = SSTdataset(opt, 'test', tokenizer)
            if opt.model_name == 'TCN':
                self.model = opt.model_class(opt, embedding_matrix, opt.input_dim, 2, [opt.input_dim]*4, dropout=0.1, emb_dropout=0.5,
                                         kernel_size=3, tied_weights=True)
            else:
                self.model = opt.model_class(opt)
        elif opt.dataset == 'sst2':
            tokenizer = BertTokenizer.from_pretrained(opt.pretrained_model)
            self.model = opt.model_class(opt)
            self.train_data = SSTdataset(opt, 'train', tokenizer)
            self.test_data = SSTdataset(opt, 'test', tokenizer)

        else:
            self.model = opt.model_class(opt)
            self.train_data = UsualDataset(opt, 'train')
            self.test_data = UsualDataset(opt, 'test')


        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=opt.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self._print_args()
    def run(self):
        train_data_loader = DataLoader(dataset=self.train_data, batch_size=self.opt.batch_size, shuffle=True, drop_last=False)
        test_data_loader = DataLoader(dataset=self.test_data, batch_size=self.opt.batch_size, shuffle=True)
        # self._reset_params()
        if self.opt.save_weight is True:
            best_model_path = self._train(train_data_loader, test_data_loader)
            self.model.load_state_dict(torch.load(best_model_path))
            self.model.eval()
            test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
            logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(9, 9))
            train_acc_list, test_acc_list = self._train(train_data_loader, test_data_loader)
            logger.info('test acc:{}'.format(max(test_acc_list)))
            ax.plot(np.arange(1, self.opt.num_epoch+1), train_acc_list)
            ax.plot(np.arange(1, self.opt.num_epoch+1), test_acc_list)
            plt.show()
    def _train(self, train_data_loader, test_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        val_acc_list = []
        train_acc_list = []
        self.model.cuda()
        for epoch in range(self.opt.num_epoch):
            self.model.train()
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch + 1))
            n_correct, n_total, loss_total = 0, 0, 0
            for batch in train_data_loader:
                global_step += 1
                self.optimizer.zero_grad()
                input, label = batch['text_indices'].cuda(), batch['polarity'].cuda()
                outputs = self.model(input)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()


                n_correct += (torch.argmax(outputs, -1) == label).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            train_acc_list.append(n_correct/n_total)
            val_acc, val_f1 = self._evaluate_acc_f1(test_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if self.opt.save_weight is True:
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    if not os.path.exists('state_dict'):
                        os.mkdir('state_dict')
                    path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.opt.model_name, self.opt.dataset,
                                                                  round(val_acc, 4))
                    torch.save(self.model.state_dict(), path)
                    logger.info('>> saved: {}'.format(path))
                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1
            else:
                val_acc_list.append(val_acc)
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    logger.info('>> best acc: {}'.format(val_acc))
                path = (train_acc_list, val_acc_list)

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                input, t_targets = t_sample_batched['text_indices'].cuda(), t_sample_batched['polarity'].cuda()
                t_outputs = self.model(input)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        acc = n_correct / n_total
        f1 = f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    # parser.add_argument('--train', action='store_false')
    # parser.add_argument('--mixed_precision', default=True, type=bool)
    parser.add_argument('--l2reg', default=0.01, type=float)
    # parser.add_argument('--embed_dim', default=300, type=int)
    # parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')


    # The following parameters are only valid for the lcf-bert model
    # parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    # parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    # GNN config
    parser.add_argument('--save_weight',action='store_true', default=False)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--input_dim', default=768, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--output_dim', default=2, type=int)
    parser.add_argument('--dataset', default='sst2', type=str)
    parser.add_argument('--model_name', default='TinyBertLSTMAttentionSST', type=str, help='try BertOnly, TinyBert')
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--num_head', default=12, type=int)
    parser.add_argument('--pool', default='mean', type=str)
    parser.add_argument('--decoder_dim', default=768, type=int)
    parser.add_argument('--num_epoch', default=15, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--seed', default=17, type=int, help='set seed for reproducibility')
    opt = parser.parse_args()

    dataset = {
        'usual': './data/usual',
        'virus': './data/virus',
        'sst2': './data/SST-2'
    }
    opt.polarity_dict = {
        'angry': 0,
        'sad': 1,
        'happy': 2,
        'fear': 3,
        'surprise': 4,
        'neural': 5,
        'neutral': 5
    }
    pretrained_models = {
        'BertOnly': "bert-base-chinese",
        'TinyBertOnly': "voidful/albert_chinese_tiny",
        'TinyBertLSTM': "voidful/albert_chinese_tiny",
        # 'TinyBertLSTMAttention': "bert-base-chinese",
        # 'TinyBertLSTMAttention': 'sentence-transformers/ce-ms-marco-TinyBERT-L-6',
        'TinyBertLSTMAttention': 'voidful/albert_chinese_tiny',
        'TinyBertCNN': 'voidful/albert_chinese_tiny',
        'TinyBertTCN': 'voidful/albert_chinese_tiny',
        'TinyBertLSTMAttentionSST': 'sentence-transformers/ce-ms-marco-TinyBERT-L-6'
    }
    model_classes = {
        'BertOnly': BertOnly,
        'TinyBertLSTM': TinyBertLSTM,
        'TinyBertOnly': TinyBertOnly,
        'TinyBertLSTMAttention': TinyBertLSTMAttention,
        'TinyBertCNN': TinyBertCNN,
        'TinyBertTCN': TinyBertTCN,
        'TCN': TCN,
        'TinyBertLSTMAttentionSST': TinyBertLSTMAttentionSST

    }
    if 'Bert' in opt.model_name:
        opt.pretrained_model = pretrained_models[opt.model_name]
    opt.model_class = model_classes[opt.model_name]
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.dataset_file = dataset[opt.dataset]
    log_file = './log/{}-{}-{}.log'.format('usual', opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
