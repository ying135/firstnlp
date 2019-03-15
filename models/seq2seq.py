import torch.nn as nn
import torch
import models

class seq2seq(nn.Module):
    def __init__(self, opt):
        super(seq2seq, self).__init__()
        self.encoder = models.rnn_encoder(opt)
        self.decoder = models.rnn_decoder(opt)

    def forward(self, src, src_len):
        # contexts, state = self.encoder(src, src_len.tolist())
        contexts, state = self.encoder(src, src_len) # try tensor (without transform to list) ok or not

