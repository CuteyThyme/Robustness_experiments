# coding=utf-8
import torch


class Config(object):
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.encoding_type = 'bio'
        self.batch_size = 16
        self.num_layers = 2
        self.hidden_size = 256
        self.lr = 1e-4
        self.optim_type = 'adam'
        self.trans_dropout = 0.45
        self.warmup_steps = 0.01
        
        self.max_length = 300
        self.rnn_hidden = 512
        self.bert_embedding = 768
        self.hidden_dim = 512
        self.dropout1 = 0.2
        self.dropout_ratio = 0.5   # BiLSTM-CRF dropout 
        self.rnn_layer = 2
        self.load_model = False
        self.base_epoch = 100
      
    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])