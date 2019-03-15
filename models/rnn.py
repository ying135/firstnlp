import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import os
import pickle


class rnn_encoder(nn.Module):
    def __init__(self, opt):
        super(rnn_encoder, self).__init__()
        embed_dim = opt.embeddim

        if opt.use_embedvector_src:
            with open(os.path.join('..', opt.root, 'embedmatrix_src.pkl'), 'rb') as f:
                embedmatri = pickle.load(f)
            embedmatrix = torch.Tensor(embedmatri)
            self.embed = nn.Embedding(opt.voca_length_src, embed_dim, _weight=embedmatrix)
        else:
            self.embed = nn.Embedding(opt.voca_length_src, embed_dim)

        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=opt.hidden_size,
                           dropout=opt.dropout)

    def forward(self, x, lengths):   # x (seq_len, batch)
        embed = pack_padded_sequence(self.embed(x), lengths)    # (seq_len, batch, embed_dim)
        output, (hn, cn) = self.rnn(x)  # output (seq_len,batch,hidden_size) hn\cn (1, batch, hidden_size)
        output = pad_packed_sequence(output)[0]
        return output, (hn, cn)


class rnn_decoder(nn.Module):
    def __init__(self, opt):
        super(rnn_decoder, self).__init__()
        embed_dim = opt.embeddim

        if opt.use_embedvector_tgt:
            with open(os.path.join('..', opt.root, 'embedmatrix_tgt.pkl'), 'rb') as f:
                embedmatri = pickle.load(f)
            embedmatrix = torch.Tensor(embedmatri)
            self.embed = nn.Embedding(opt.voca_length_tgt, embed_dim, _weight=embedmatrix)
        else:
            self.embed = nn.Embedding(opt.voca_length_tgt, embed_dim)

        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=opt.hidden_size,
                           dropout=opt.dropout)
        self.linear = nn.Linear(opt.hidden_size, opt.voca_length_tgt)

    def forward(self, x):
        pass

