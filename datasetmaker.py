import torch
from torch.utils import data


# many to many, the first edition of this class is based on en_vi translation
class m2mdata(data.Dataset):
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
