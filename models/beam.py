import torch
import dict


class Beam(object):
    def __init__(self, size, use_cuda, n_best=1):
        self.size = size
        self.scores = torch.zeros(size).float()
        self.allScores = []

        self.prevKs = []

        self.nextYs = [torch.LongTensor(size).fill_(dict.EOS)]
        self.nextYs[0][0] = dict.BOS

        self._eos = dict.EOS
        self.eosTop = False

        self.attn = []

        self.finished = []  # (score, length-1, ith beam)
        self.n_best = n_best

        if use_cuda:
            self.scores.cuda()
            self.nextYs.cuda()

    def getCurrentState(self):
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut):
        numWords = wordLk.size(1)   # vocalength
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.allScores.append(self.scores)
        self.scores = bestScores

        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        self.attn.append(attnOut.index_select(0, prevK))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == dict.EOS:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.n_best

    def beam_update(self, state, idx):
        # state h/c (1, batch * beam_size, hiddensize)
        positions = self.getCurrentOrigin()
        for e in state:
            a, br, d = e.size()
            e = e.view(a, self.size, br // self.size, d)
            senStates = e[:, :, idx]
            senStates.data.copy_(senStates.index_select(1, positions))

    def sortFinished(self, minimum=None):
        if minimum is not None:
            i = 0
            while len(self.finished) < minimum:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))
        self.finished.sort(key=lambda a:-a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def getHyp(self, timestep, k):
        hyp, attn = [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]
        return hyp[::-1], torch.stack(attn[::-1])
