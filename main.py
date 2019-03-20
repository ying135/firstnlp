import torch
from torch.utils.data import DataLoader
import config
import pickle
import os
import datasetmaker
import models
import dict
import time
import utils
import codecs
import optims
import metrics

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


def train(model, trainloader, validloader, params):
    if opt.use_cuda:
        model = model.cuda()
    model.train()
    # mine is so easy, copy deconv's optim
    # optimizer = optims.Optim('adam', opt.lr, opt.max_grad_norm, opt.lr_decay, opt.start_decay_at)
    # optimizer.set_parameters(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    previous_loss = 1e10
    pEpoch = []
    pLoss = []
    iter_all = 0
    total_loss, total_num = 0, 0
    for epoch in range(opt.epoch):
        for i, (input, target, src_len, tgt_len, inputstr, targetstr) in enumerate(trainloader):
            if opt.use_cuda:
                input = input.cuda()
                target = target.cuda()
                src_len = src_len.cuda()
                tgt_len = tgt_len.cuda()
            # optimizer.optimizer.zero_grad()   # not sure about this , usually optimizer.zreo_grad()
            optimizer.zero_grad()

            try:
                outputs, targets = model(input, src_len, target)
                loss, num_total, num_correct = model.compute_loss(outputs, targets)

                loss = torch.sum(loss) / num_total
                loss.backward()
                optimizer.step()

                params['report_loss'] += float(loss)
                params['report_correct'] += num_correct
                params['report_total'] += num_total
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

            utils.progress_bar(params['updates'], opt.val_inter)
            params['updates'] += 1

            if params['updates'] % opt.val_inter == 0:
                params['log']("epoch: %3d, loss: %6.3f, time: %6.3f, updates: %8d, accuracy: %2.2f\n"
                              % (epoch, params['report_loss'], time.time() - params['report_time'],
                                 params['updates'], params['report_correct'] * 100.0 / params['report_total']))
                print('evaluating after %d updates...\r' % params['updates'])
                score = valid(model, validloader, params)
                for metric in opt.metrics:
                    params[metric].append(score[metric])
                    if score[metric] >= max(params[metric]):
                        with codecs.open(params['log_path'] + 'best_' + metric + '_prediction.txt', 'w', 'utf-8') as f:
                            f.write(codecs.open(params['log_path'] + 'candidate.txt', 'r', 'utf-8').read())
                        # save_model(params['log_path'] + 'best_' + metric + '_checkpoint.pt', model, optim,
                        #            params['updates'])
                params['report_loss'], params['report_time'] = 0, time.time()
                params['report_correct'], params['report_total'] = 0, 0



        # if params['updates'] % config.save_interval == 0:
        #     save_model(params['log_path'] + 'checkpoint.pt', model, optim, params['updates'])

        # update lr
        # optimizer.updateLearningRate(score=0, epoch=epoch)


def valid(model, validloader, params):
    model.eval()
    candidate, source, reference, alignments = [], [], [], []
    for i, (input, target, src_len, tgt_len, inputstr, targetstr) in enumerate(validloader):
        if opt.use_cuda:
            input = input.cuda()
            target = target.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()

        with torch.no_grad():
            if opt.beam_size > 1:
                samples, alignment = model.beam_sample(input, src_len, opt.beam_size)

        candidate += [opt.dict_tgt.idx2words(s, dict.EOS) for s in samples]
        source += inputstr
        reference += targetstr
        alignments += [align for align in alignment]
        utils.progress_bar(i, len(validloader))
        # for i in range(len(candidate)):
        #     print(source[i])
        #     print(reference[i])
        #     print(candidate[i])

    if opt.replace_unk:
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == dict.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
            if len(cand) == 0:
                print('Error!')
        candidate = cands

    with codecs.open(params['log_path'] + 'candidate.txt', 'w+', 'utf-8') as f:
        for i in range(len(candidate)):
            f.write(" ".join(candidate[i]) + '\n')

    score = {}
    for metric in opt.metrics:
        score[metric] = getattr(metrics, metric)(reference, candidate, params['log_path'], params['log'], opt)
    model.train()
    return score


def test(model, testloader):
    pass


def load_model():
    model = models.seq2seq(opt)
    if opt.load_model_path:
        model.load_state_dict(torch.load(opt.load_model_path))
        print("Load Success!", opt.load_model_path)
    return model




def main():
    if opt.train:
        trainloader, validloader, dict_src, dict_tgt = load_data(opt.train)
        opt.parse({'voca_length_src': len(dict_src), 'voca_length_tgt': len(dict_tgt),
                   'dict_tgt':dict_tgt})
        model = load_model()
        print_log, log_path = utils.build_log()
        params = {'updates': 0, 'report_loss': 0, 'report_total': 0,
                  'report_correct': 0, 'report_time': time.time(),
                  'log': print_log, 'log_path': log_path}
        for metric in opt.metrics:
            params[metric] = []
        print("START training...")
        train(model, trainloader, validloader, params)
    else:
        testloader, dict_src, dict_tgt = load_data(opt.train)
        opt.parse({'voca_length_src': len(dict_src), 'voca_length_tgt': len(dict_tgt),
                   'dict_tgt':dict_tgt})
        model = load_model()
        print("START testing...")
        test(model, testloader)


if __name__ == "__main__":
    main()
