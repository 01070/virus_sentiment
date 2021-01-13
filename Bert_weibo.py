import torch
import torch.nn as nn
from tcn import TemporalConvNet
from transformers import BertModel, AutoModel
import torch.nn.functional as F

class BertOnly(nn.Module):
    def __init__(self, opt):
        super(BertOnly, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dense = nn.Linear(768, 6)
        self.dropout = nn.Dropout(0.1)
    def forward(self, inputs):
        outputs = self.bert(input_ids=inputs)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits

class TinyBertOnly(nn.Module):
    def __init__(self, opt):
        super(TinyBertOnly, self).__init__()
        self.bert = AutoModel.from_pretrained("voidful/albert_chinese_tiny")
        self.dense = nn.Linear(312, 6)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        outputs = self.bert(input_ids=inputs)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits


class TinyBertLSTM(nn.Module):
    def __init__(self, opt):
        super(TinyBertLSTM, self).__init__()
        self.bert = AutoModel.from_pretrained("voidful/albert_chinese_tiny")
        self.bilstm = nn.LSTM(312, 128, 1, bidirectional=True)
        self.dense1 = nn.Linear(312, 256)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.dense2 = nn.Linear(256, 6)
    def init_hidden(self, batch_size):
        return torch.randn(2, batch_size, 128).cuda(), torch.randn(2, batch_size, 128).cuda()

    def forward(self, inputs):
        outputs = self.bert(input_ids=inputs)
        hidden_x, pooled_output = outputs
        h0, c0 = self.init_hidden(hidden_x.size(0))
        lstm_out_x, (h, c) = self.bilstm(hidden_x.permute(1, 0, 2), (h0.cuda(), c0.cuda()))
        logits = self.dense2(F.relu(lstm_out_x[-1]))


        return logits


class TinyBertLSTMAttention(nn.Module):
    def __init__(self, opt):
        super(TinyBertLSTMAttention, self).__init__()
        self.bert = AutoModel.from_pretrained("voidful/albert_chinese_tiny")
        self.bilstm = nn.LSTM(opt.input_dim, 128, 1, bidirectional=True)
        self.dense1 = nn.Linear(opt.input_dim, 256)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0)
        self.dense2 = nn.Linear(256, opt.output_dim)
    def init_hidden(self, batch_size):
        return torch.randn(2, batch_size, 128).cuda(), torch.randn(2, batch_size, 128).cuda()

    def forward(self, inputs):
        outputs = self.bert(input_ids=inputs)
        hidden_x, pooled_output = outputs
        h0, c0 = self.init_hidden(hidden_x.size(0))
        lstm_out_x, (h, c) = self.bilstm(hidden_x.permute(1, 0, 2), (h0.cuda(), c0.cuda()))
        pool_drop = self.dense1(self.dropout(pooled_output))
        a = self.softmax(torch.matmul(pool_drop.view(-1, 1, 256), lstm_out_x.permute(1, 2, 0)))
        a_vec = torch.matmul(a, lstm_out_x.permute(1, 0, 2))
        a_vec_dropout = self.dropout(F.relu(a_vec.view(-1, 256)))
        logits = self.dense2(a_vec_dropout)
        return logits


class TinyBertCNN(nn.Module):
    def __init__(self, opt):
        super(TinyBertCNN, self).__init__()
        self.bert = AutoModel.from_pretrained("voidful/albert_chinese_tiny")
        self.kernel = 2
        self.layer1 = nn.Sequential(nn.Conv1d(312, self.kernel, 3),
                                    nn.MaxPool1d(3),
                                    nn.ReLU()
                                    )

        self.layer2 = nn.Sequential(nn.Conv1d(312, self.kernel, 4),
                                    nn.MaxPool1d(3),
                                    nn.ReLU()
                                    )

        self.layer3 = nn.Sequential(nn.Conv1d(312, self.kernel, 5),
                                    nn.MaxPool1d(3),
                                    nn.ReLU()
                                    )

        self.maxpool = nn.MaxPool1d(3)
        self.dense = nn.Linear(128, 6)

    def forward(self, inputs):
        layers = [self.layer1, self.layer2, self.layer3]
        hidden_x, pooled_output = self.bert(input_ids=inputs)
        x1, x2, x3 = [layer(hidden_x.permute(0, 2, 1)) for layer in layers]

        max_out4 = F.relu(torch.cat((self.maxpool(x1), self.maxpool(x2), self.maxpool(x3)), dim=-1))
        logits = self.dense(max_out4.view(-1, 128))
        return logits


class TinyBertTCN(nn.Module):
    def __init__(self, opt, embedding_matrix, input_size, output_size, num_channels,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):
        super(TinyBertTCN, self).__init__()
        self.output_size = output_size
        self.bert = AutoModel.from_pretrained("voidful/albert_chinese_tiny")
        self.tcn = TemporalConvNet(312, num_channels, kernel_size, dropout=dropout)
        self.dense = nn.Linear(num_channels[-1], output_size)
        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            # self.dense.weight = self.decode.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        # self.init_weights()

    # def init_weights(self):
    #     self.encoder.weight.data.normal_(0, 0.01)
    #     self.decoder.bias.data.fill_(0)
    #     self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, inputs):
        hidden_x, pooled_output = self.bert(input_ids=inputs)
        y = self.tcn(hidden_x.permute(0, 2, 1))
        y = self.dense(y.permute(0, 2, 1))
        logits = y[:, -1].view(-1, self.output_size)
        return logits.contiguous()


class TCN(nn.Module):
    def __init__(self, opt, embedding_matrix, input_size, output_size, num_channels,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):
        super(TCN, self).__init__()
        self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.output_size = output_size
        self.tcn = TemporalConvNet(opt.input_dim, num_channels, kernel_size, dropout=dropout)
        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder = nn.Linear(num_channels[-1], opt.decoder_dim)
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.mhsa = nn.MultiheadAttention(opt.decoder_dim, opt.num_head)
        self.lin_q = nn.Linear(opt.decoder_dim, opt.decoder_dim)
        self.lin_k = nn.Linear(opt.decoder_dim, opt.decoder_dim)
        self.lin_v = nn.Linear(opt.decoder_dim, opt.decoder_dim)
        self.emb_dropout = emb_dropout
        self.relu = nn.ReLU()
        self.pool_way = opt.pool
        self.init_weights()
        self.dense = nn.Linear(opt.decoder_dim, output_size)

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, inputs):

        hidden_x = self.encoder(inputs)
        x = self.tcn(hidden_x.permute(0, 2, 1))
        x = self.decoder(x.permute(0, 2, 1))
        query = self.lin_q(x.permute(1, 0, 2))
        key = self.lin_k(x.permute(1, 0, 2))
        value = self.lin_v(x.permute(1, 0, 2))
        output, _ = self.mhsa(query, key, value)
        if self.pool_way == 'max':
            output = torch.max(output, dim=0)[0]
        elif self.pool_way == 'mean':
            output = torch.mean(output, dim=0)
        else:
            output = (torch.mean(output, dim=0) + torch.max(output, dim=0)[0])/2
        logits = self.dense(self.relu(output))
        return logits.contiguous()

class TinyBertLSTMAttentionSST(nn.Module):
    def __init__(self, opt):
        super(TinyBertLSTMAttentionSST, self).__init__()
        self.bert = AutoModel.from_pretrained("sentence-transformers/ce-ms-marco-TinyBERT-L-6")
        self.bilstm = nn.LSTM(opt.input_dim, 128, 1, bidirectional=True)
        self.dense1 = nn.Linear(opt.input_dim, 256)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0)
        self.dense2 = nn.Linear(256, opt.output_dim)
    def init_hidden(self, batch_size):
        return torch.randn(2, batch_size, 128).cuda(), torch.randn(2, batch_size, 128).cuda()

    def forward(self, inputs):
        outputs = self.bert(input_ids=inputs)
        hidden_x, pooled_output = outputs
        h0, c0 = self.init_hidden(hidden_x.size(0))
        lstm_out_x, (h, c) = self.bilstm(hidden_x.permute(1, 0, 2), (h0.cuda(), c0.cuda()))
        pool_drop = self.dense1(self.dropout(pooled_output))
        a = self.softmax(torch.matmul(pool_drop.view(-1, 1, 256), lstm_out_x.permute(1, 2, 0)))
        a_vec = torch.matmul(a, lstm_out_x.permute(1, 0, 2))
        a_vec_dropout = self.dropout(F.relu(a_vec.view(-1, 256)))
        logits = self.dense2(a_vec_dropout)
        return logits