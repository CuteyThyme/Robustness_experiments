import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size=100, num_layers=1, dropout=0.0, batch_first=True,
                 bidirectional=False, bias=True):
        super(LSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)
        self.init_param()

    def init_param(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, seq_len=None, h0=None, c0=None):
        batch_size, max_len, _ = x.size()
        if h0 is not None and c0 is not None:
            hx = (h0, c0)
        else:
            hx = None
        if seq_len is not None and not isinstance(x, rnn.PackedSequence):
            sort_lens, sort_idx = torch.sort(seq_len, dim=0, descending=True)
            if self.batch_first:
                x = x[sort_idx]
            else:
                x = x[:, sort_idx]
            x = rnn.pack_padded_sequence(x, sort_lens, batch_first=self.batch_first)
            output, hx = self.lstm(x, hx)  # -> [N,L,C]
            output, _ = rnn.pad_packed_sequence(output, batch_first=self.batch_first, total_length=max_len)
            _, unsort_idx = torch.sort(sort_idx, dim=0, descending=False)
            if self.batch_first:
                output = output[unsort_idx]
            else:
                output = output[:, unsort_idx]
            hx = hx[0][:, unsort_idx], hx[1][:, unsort_idx]
        else:
            output, hx = self.lstm(x, hx)
        return output, hx
