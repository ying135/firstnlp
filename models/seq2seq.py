import torch.nn as nn
import torch
import torch.nn.functional as F
import models


class seq2seq(nn.Module):
    def __init__(self, opt):
        super(seq2seq, self).__init__()
        self.opt = opt
        self.encoder = models.rnn_encoder(opt)
        self.decoder = models.rnn_decoder(opt)
        self.criterion = models.criterion(opt.voca_length_tgt, opt.use_cuda)
        self.log_softmax = nn.LogSoftmax(dim=1)

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
        _, ind = torch.sort(indices)
        src = torch.index_select(src, dim=1, index=indices)
        contexts, encState = self.encoder(src, lengths.tolist())
        # [guess] didn't write contexts.data
        contexts = rep(contexts).transpose(0, 1)    # (batch*beam_size, seq_len_src, hiddensize)
        decState = (rep(encState[0]), rep(encState[1])) # (1,batch*beam_size,hiddensize)
        beam = [models.Beam(beam_size, self.opt.use_cuda) for __ in range(batch_size)]

        for i in range(self.opt.max_tgt_len):
            if all((b.done() for b in beam)):
                break

            inp = torch.stack([b.getCurrentState() for b in beam]).t().contiguous().view(-1)
            # (beam_size * batch)

            output, decState, attn = self.decoder.sample_one(inp, decState, contexts)
            soft_score = F.softmax(output, dim=1)  # (beamsize*batch, vocalength)
            predicted = output.max(1)[1]
            mask = predicted.unsqueeze(1).long()    # (beamsize*batch, 1)

            output = self.log_softmax(output).view(beam_size, batch_size, -1)
            # (beamsize, batch, vocalength)     make them negative,why??
            attn = attn.view(beam_size, batch_size, -1) # (beam_size,batch, seq_len_src)

            for j, b in enumerate(beam):
                b.advance(output[:, j], attn[:, j])
                b.beam_update(decState, j)

        allHyps, allScores, allAttn = [], [], []

        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])

        return allHyps, allAttn

