from torch import nn

from fastNLP.modules.encoder.seq2seq_encoder import TransformerSeq2SeqEncoderLayer


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, d_model=512, n_head=8, dim_ff=2048, dropout=0.1):
    
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerSeq2SeqEncoderLayer(d_model = d_model, n_head = n_head, dim_ff = dim_ff,
                 dropout = dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, seq_mask=None):
        output = x
        if seq_mask is None:
            seq_mask = x.new_ones(x.size(0), x.size(1)).bool()
        for layer in self.layers:
            output = layer(output, seq_mask)
        return self.norm(output)
