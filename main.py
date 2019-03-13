import torch
from torch.utils.data import DataLoader
import config
import pickle
import os
import datasetmaker
import models

opt = config.DefaultConfig()


def load_data(train):
    print("loading data...")
    with open(os.path.join(opt.root, 'data.pkl'), 'rb') as f:
        data = pickle.load(f)
    dict_src = data['dicts']['src']
    dict_tgt = data['dicts']['tgt']

    if train:
        data['train']['length'] = int(data['train']['length'] * opt.data_ratio)
        data['valid']['length'] = int(data['valid']['length'] * opt.data_ratio)
        trainset = datasetmaker.m2mdata(data['train'])
        validset = datasetmaker.m2mdata(data['valid'])
        trainloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                 num_workers=opt.num_worker, collate_fn=datasetmaker.padding)
        validloader = DataLoader(validset, batch_size=opt.batch_size, shuffle=True,
                                 num_workers=opt.num_worker, collate_fn=datasetmaker.padding)
        # the loader generate the batch which contains: src_pad, tgt_pad, src_len, tgt_len, srcstr, tgtstr
        # src_pad and tgt_pad dimension : (batch_size, max(src_len) in this batch)
        # src_len: length of sentence(after being filtered and trunc)  dimension: (batchsize)
        # tgt_len: length of sentence(after being filtered and trunc) plus 2(bos eos) dimension: (batchsize)
        # srcstr, tgtstr : the string with no padding
        return trainloader, validloader, dict_src, dict_tgt
    else:
        data['test']['length'] = int(data['test']['length'] * opt.data_ratio)
        testset = datasetmaker.m2mdata(data['test'])
        testloader = DataLoader(testset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_worker)
        return testloader, dict_src, dict_tgt


def train(model, trainloader, validloader):
    if opt.use_cuda:
        model = model.cuda()
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=opt.weight_decay)
    previous_loss = 1e10
    pEpoch = []
    pLoss = []

    for epoch in range(opt.epoch):

        for i, (input, target, src_len, tgt_len, inputstr, targetstr) in enumerate(trainloader):
            if opt.use_cuda:
                input = input.cuda()
                taget = target.cuda()
                src_len = src_len.cuda()
                tgt_len = tgt_len.cuda()
            optimizer.zero_grad()




def valid(model, validloader):
    pass


def test(model, testloader):
    pass


def main():
    model = models.seq2seq()
    if opt.load_model_path:
        model.load_state_dict(torch.load(opt.load_model_path))
        print("Load Success!", opt.load_model_path)

    if opt.train:
        trainloader, validloader, dict_src, dict_tgt = load_data(opt.train)
        print("START training...")
        train(model, trainloader, validloader)
    else:
        testloader, dict_src, dict_tgt = load_data(opt.train)
        print("START testing...")
        test(model, testloader)


if __name__ == "__main__":
    main()
