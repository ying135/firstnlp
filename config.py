import torch


class DefaultConfig(object):
    train = True
    load_model_path = None
    data_ratio = 1  # the proportion of trainset
    use_embedvector_src = False
    use_embedvector_tgt = False
    embeddim = 100
    voca_length_src = 123  # 123 is written randomly, its value will be modified in the main.py
    voca_length_tgt = 456
    dict_tgt = None     # will be updated in the main.py

    epoch = 200
    batch_size = 16
    num_worker = 0
    lr = 0.001
    weight_decay = 0e-5
    lr_decay = 0.5
    attention = 'dot'

    print_inter = 20
    val_inter = 2000

    beam_size = 1
    max_tgt_len = 25
    hidden_size_encoder = 256
    hidden_size_decoder = 256
    dropout = 0

    use_cuda = torch.cuda.is_available()
    root = 'data'
    metrics = ['bleu']

    max_grad_norm = 15
    start_decay_at = 3

    replace_unk = True
    char_tgt = False

    def parse(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
