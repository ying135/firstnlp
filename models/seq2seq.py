import torch.nn as nn
import torch
import models

class seq2seq(nn.Module):
    def __init__(self, opt):
        super(seq2seq, self).__init__()
        self.encoder = models.rnn_encoder(opt)
        self.decoder = models.rnn_decoder(opt)
        self.criterion = models.criterion(opt.voca_length_tgt, opt.use_cuda)

    def forward(self, src, src_len, tgt):
        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        src = torch.index_select(src, dim=1, index=indices)
        tgt = torch.index_select(tgt, dim=1, index=indices)

        contexts, state = self.encoder(src, lengths.tolist())
        # contexts, state = self.encoder(src, src_len) # try tensor (without transform to list) ok or not
        outputs, final_state = self.decoder(tgt[:-1], state, contexts.transpose(0, 1))
        return outputs, tgt[1:]
        # (tgt.size(0)-1, batch, hidden_size)

    def compute_loss(self, outputs, tgt):
        return models.cross_entropy_loss(outputs, self.decoder, tgt, self.criterion)

