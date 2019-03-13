import torch


class DefaultConfig(object):
    train = True
    load_model_path = None
    data_ratio = 1 # the proportion of trainset

    epoch = 100
    batch_size = 64
    num_worker = 4
    lr = 0.001
    weight_decay = 0e-5
    lr_decay = 0.5
    print_inter = 20
    hidden_size = 256

    use_cuda = torch.cuda.is_available()
    root = 'data'


    def parse(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
