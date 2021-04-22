import torch
import torch.nn as nn
import torch.nn.functional as F

from fastNLP.models.base_model import BaseModel
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.core.const import Const as C
from fastNLP import CrossEntropyLoss

from layers.lstm import LSTM
from layers.transformer import TransformerEncoder
from layers.crf import ConditionalRandomField, allowed_transitions

from config import Config

config = Config()

class NERModel(BaseModel):
    
    def __init__(self, embed, num_classes, num_layers=2, hidden_size=100, dropout=0.1, encoder='lstm', decoder='crf',
                  target_vocab=None):
        super().__init__()
        self.embed = get_embeddings(embed)
        self.encoder = encoder
        self.decoder = decoder

        if encoder == 'lstm':
            self.lstm = LSTM(self.embed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size, bidirectional=True,
                             batch_first=True, dropout=dropout)
        elif encoder == 'cnn':
            self.word2cnn = nn.Linear(self.embed.embedding_dim, hidden_sizse*2)
            self.cnn_list = list()
            for _ in range(4):
                self.cnn_list.append(nn.Conv1d(hidden_size*2, hidden_size*2, kernel_size=3, padding=1))
                self.cnn_list.append(nn.ReLU())
                self.cnn_list.append(nn.Dropout(dropout))
                self.cnn_list.append(nn.BatchNorm1d(hidden_size*2))
            self.cnn = nn.Sequential(*self.cnn_list)
        elif encoder == 'transformer':
            self.transformer = TransformerEncoder()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)

        trans = None
        if target_vocab is not None:
            assert len(target_vocab)==num_classes, "The number of classes should be same with the length of target vocabulary."
            trans = allowed_transitions(target_vocab.idx2word, include_start_end=True)

        self.crf = ConditionalRandomField(num_classes, include_start_end_trans=True, allowed_transitions=trans)
       
    def _forward(self, words, seq_len=None, target=None):
        batch_size = config.batch_size
        words = self.embed(words)

        if self.encoder == 'lstm':
            feats, _ = self.lstm(words, seq_len=seq_len)
        elif self.encoder == 'cnn':
            word_in = torch.tanh(self.word2cnn(words)).transpose(2, 1).contiguous()
            feats = self.cnn(word_in).transpose(1, 2).contiguous()
        elif self.encoder == 'transformer':
            feats = self.transformer(words)

        feats = self.fc(feats)
        feats = self.dropout(feats)
        
        if self.decoder == 'crf':
            logits = F.log_softmax(feats, dim=-1)
            mask = seq_len_to_mask(seq_len)
        
            if target is None:
                pred, _ = self.crf.viterbi_decode(logits, mask)
                return {C.OUTPUT:pred}
            else:
                loss = self.crf(logits, target, mask).mean()
                return {C.LOSS:loss}
        else:
            feature_out = feats.contiguous().view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(feature_out, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq
            
            if target is None:
                return {C.OUTPUT: pred}
            else:
                loss = CrossEntropyLoss(pred=feature_out, target=target, seq_len=seq_len, reduction='sum')
                return {C.LOSS: loss}

    def forward(self, words, seq_len, target):
        return self._forward(words, seq_len, target)

    def predict(self, words, seq_len):
        return self._forward(words, seq_len)


