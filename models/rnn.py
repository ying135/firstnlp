import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import os
import pickle
import models


class rnn_encoder(nn.Module):
    def __init__(self, opt):
        super(rnn_encoder, self).__init__()
        embed_dim = opt.embeddim
        hidden_size = opt.hidden_size_encoder
        voca_length = opt.voca_length_src

        if opt.use_embedvector_src:
            with open(os.path.join('..', opt.root, 'embedmatrix_src.pkl'), 'rb') as f:
                embedmatri = pickle.load(f)
            embedmatrix = torch.Tensor(embedmatri)
            self.embed = nn.Embedding(voca_length, embed_dim, _weight=embedmatrix)
        else:
            self.embed = nn.Embedding(voca_length, embed_dim)

        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                           dropout=opt.dropout)

    def forward(self, x, lengths):   # x (seq_len, batch)
        embed = pack_padded_sequence(self.embed(x), lengths)    # (seq_len, batch, embed_dim)
        output, (hn, cn) = self.rnn(embed)  # output (seq_len,batch,hidden_size) hn\cn (1, batch, hidden_size)
        output = pad_packed_sequence(output)[0]
        return output, (hn, cn)


class rnn_decoder(nn.Module):
    def __init__(self, opt):
        super(rnn_decoder, self).__init__()
        embed_dim = opt.embeddim
        hidden_size = opt.hidden_size_decoder
        voca_length = opt.voca_length_tgt

        if opt.use_embedvector_tgt:
            with open(os.path.join('..', opt.root, 'embedmatrix_tgt.pkl'), 'rb') as f:
                embedmatri = pickle.load(f)
            embedmatrix = torch.Tensor(embedmatri)
            self.embed = nn.Embedding(voca_length, embed_dim, _weight=embedmatrix)
        else:
            self.embed = nn.Embedding(voca_length, embed_dim)

        self.opt = opt
        # my rnn is different with yang, his input will be 1 dim, mine is 2 dim
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                           dropout=opt.dropout)
        self.linear = nn.Linear(hidden_size, voca_length)
        self.dropout = nn.Dropout(opt.dropout)
        self.attention = models.global_attention(hidden_size)

    def forward(self, x, init_state, contexts):
        # contexts (batch, seq_len, hidden_size)
        embeds = self.embed(x)
        outputs, state, attns = [], init_state, []
        for emb in embeds.split(1):
            # output, state = self.rnn(emb.squeeze(0), state)
            # I guess, yang's rnn is stacked LSTM, so need squeeze
            output, state = self.rnn(emb, state) # output (1, batch, hidden_size)
            output, attn_weights = self.attention(output, contexts) # (batch, hidden_size),(batch, seq_len_src)
            # output = self.dropout(output)
            outputs += [output]
            attns += [attn_weights]
        outputs = torch.stack(outputs)  # guess (x.size(0), batch, hidden_size) x.size(0) is seq_len_tgt-1(1 is <eos>)
        attns = torch.stack(attns)
        return outputs, state

    # don't know why, just do as yang
    def compute_score(self, outputs):
        scores = self.linear(outputs)
        return scores

    def sample_one(self, input, state, contexts):
        emb = self.embed(input).unsqueeze(0)    # (1, beam_size * batch, embeddim)
        output, state = self.rnn(emb, state)
        hidden, attn_weights = self.attention(output, contexts)
        # (beam_size *batch, hidden_size),(beam_size *batch, seq_len_src)
        output = self.compute_score(hidden) # (batch, vocalength)

        return output, state, attn_weights


