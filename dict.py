import torch

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class Dict(object):
    def __init__(self, data=None, lower=True):
        self.lower = lower
        self.word2idx = {}
        self.idx2word = {}
        self.special = []
        self.frequencies = {}
        if data is not None:
            if type(data) == str:
                self.loadfile(data)
            else:
                for word in data:
                    self.add(word)

    # not sure about this function's name, deconv name it size, not __len__
    def __len__(self):
        return len(self.idx2word)

    def loadfile(self,file):
        pass

    def writefile(self, file):
        with open(file, 'w', encoding='utf8') as f:
            for i in range(self.__len__()):
                word = self.idx2word[i]
                f.write('%s %d\n' % (word, i))

    def addspecial(self, word, idx=None):
        idx = self.add(word, idx)
        self.special += [idx]

    def add(self, word, idx=None):
        word = word.lower() if self.lower else word

        if word in self.word2idx:
            idx = self.word2idx[word]
        else:
            if idx is None:
                idx = len(self.idx2word)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1
        return idx

    def lookup(self, key, defau=None):
        key = key.lower() if self.lower else key
        try:
            return self.word2idx[key]
        except KeyError:
            return  defau

    def words2idx(self, words, unkword, bosword=None, eosword=None):
        idxVec = []
        if bosword is not None:
            idxVec += [self.lookup(bosword)]
        unk = self.lookup(unkword)
        idxVec += [self.lookup(word, unk) for word in words]
        if eosword is not None:
            idxVec += [self.lookup(eosword)]
        return idxVec

