import torch.nn as nn
import torch
import models


class seq2seq(nn.Module):
    def __init__(self, opt):
        super(seq2seq, self).__init__()
        self.opt = opt
        self.encoder = models.rnn_encoder(opt)
        self.decoder = models.rnn_decoder(opt)
        self.criterion = models.criterion(opt.voca_length_tgt, opt.use_cuda)

    def forward(self, src, src_len, tgt):
        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        src = torch.index_select(src, dim=1, index=indices)
        tgt = torch.index_select(tgt, dim=1, index=indices)

        # i test that, if not transform lengths to list   still work
        contexts, state = self.encoder(src, lengths.tolist())
        outputs, final_state = self.decoder(tgt[:-1], state, contexts.transpose(0, 1))
        return outputs, tgt[1:]
        # (tgt.size(0)-1, batch, hidden_size)

    def compute_loss(self, outputs, tgt):
        return models.cross_entropy_loss(outputs, self.decoder, tgt, self.criterion)

    def beam_sample(self, src, src_len, beam_size=1):
        # pytorch tutorial say it's useful, i didn't check
        torch.set_grad_enabled(False)

        def rep(x):
            return x.repeat(1, beam_size, 1)

        batch_size = src.size(1)
        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        src = torch.index_select(src, dim=1, index=indices)
        contexts, encState = self.encoder(src, lengths.tolist())
        # [guess] didn't write contexts.data
        contexts = rep(contexts).transpose(0, 1)    # (batch*beam_size, seq_len, hiddensize)
        decState = (rep(encState[0]), rep(encState[1]))
        beam = [models.Beam(beam_size) for __ in range(batch_size)]



