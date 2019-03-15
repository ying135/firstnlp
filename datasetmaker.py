import torch
from torch.utils import data
import linecache


# many to many, the first edition of this class is based on en_vi translation
class m2mdata(data.Dataset):
    def __init__(self, roots):
        self.srcindf = roots['srcindf']
        self.tgtindf = roots['tgtindf']
        self.srcstrf = roots['srcstrf']
        self.tgtstrf = roots['tgtstrf']
        self.length = roots['length']

    def __getitem__(self, item):
        item = item + 1
        srcind = list(map(int, linecache.getline(self.srcindf, item).strip().split()))
        tgtind = list(map(int, linecache.getline(self.tgtindf, item).strip().split()))
        # didn't consider that whether char level
        srcstr = linecache.getline(self.srcstrf, item).strip().split()
        tgtstr = linecache.getline(self.tgtstrf, item).strip().split()

        return srcind, tgtind, srcstr, tgtstr

    def __len__(self):
        return self.length


def padding(data):
    srcind, tgtind, srcstr, tgtstr = zip(*data)

    src_len = [len(s) for s in srcind]
    src_pad = torch.zeros(len(srcind), max(src_len)).long()
    for i, s in enumerate(srcind):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s[:end])

    tgt_len = [len(s) for s in tgtind]
    # it's not the length of original tgt sentence, it's plus 2, bos and eos
    tgt_pad = torch.zeros(len(tgtind), max(tgt_len)).long()
    for i, s in enumerate(tgtind):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s[:end])

    return src_pad.t(), tgt_pad.t(), torch.LongTensor(src_len), \
           torch.LongTensor(tgt_len), srcstr, tgtstr


# many to one
class m2odata(data.Dataset):
    def __init__(self, root, train=True, test=False):
        self.root = root
        self.train = train
        self.test = test
        if self.test:
            self.train = False

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
