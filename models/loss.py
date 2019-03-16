import torch
import dict


def criterion(voca_length, usecuda):
    weight = torch.ones(voca_length)
    weight[dict.PAD] = 0
    crit = torch.nn.CrossEntropyLoss(weight, size_average=False)
    if usecuda:
        crit = crit.cuda()
    return crit


def cross_entropy_loss(hidden_outputs, decoder, targets, criterion):
    outputs = hidden_outputs.view(-1, hidden_outputs.size(2))   # (targets.size(0)*batch, hidden_size)
    scores = decoder.compute_score(outputs)     # (targets.size(0)*batch, voca_length_tgt)
    targets = targets.view(-1)
    loss = criterion(scores, targets)
    pred = torch.max(scores, 1)[1]
    # feel the code below will be a bug
    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(dict.PAD).data).sum()
    num_total = targets.ne(dict.PAD).data.sum()
    # [guess]didn't change the data inside loss?so it can backward?
    loss.div(num_total.float()).backward()
    loss = float(loss)
    return loss, num_total, num_correct


