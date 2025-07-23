import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from math import ceil

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    # def __init__(self, input_dim, hidden_dim, layer_dim, factor, d_ff, n_heads, output_dim):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.enc_in = 1
        self.seq_len = self.args.seq_len
        self.pred_len = self.args.pred_len
        self.hidden_dim = self.args.hidden_dim
        self.layer_dim = self.args.layer_dim
        self.seg_len = 12
        self.win_size = 2
        self.factor = self.args.factor
        self.d_ff = self.args.d_ff
        self.dropout = 0.1
        self.n_heads = self.args.n_heads
        self.output_attention = False
        
        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * self.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (self.layer_dim - 1)))
        self.head_nf = self.hidden_dim * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(self.hidden_dim, self.seg_len, self.seg_len, self.pad_in_len - self.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.enc_in, self.in_seg_num, self.hidden_dim))
        self.pre_norm = nn.LayerNorm(self.hidden_dim)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block( 1 if l == 0 else self.win_size, self.hidden_dim, self.n_heads, self.output_attention, self.d_ff,
                            1, self.dropout,
                            self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l), self.factor
                            ) for l in range(self.layer_dim)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.enc_in, (self.pad_out_len // self.seg_len), self.hidden_dim))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer((self.pad_out_len // self.seg_len), self.factor, self.hidden_dim, self.n_heads, self.output_attention,
                                           self.d_ff, self.dropout),
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=False),
                         self.hidden_dim, self.n_heads),
                    self.seg_len,
                    self.hidden_dim,
                    self.d_ff,
                    dropout=self.dropout,
                    # activation=configs.activation,
                )
                for l in range(self.layer_dim + 1)
            ],
        )


    def forward(self, x_enc, dec_inp):
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]