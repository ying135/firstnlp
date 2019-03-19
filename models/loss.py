import torch
import dict


def criterion(usecuda):
    crit = torch.nn.CrossEntropyLoss(ignore_index=dict.PAD, reduction='none')
    if usecuda:
        crit = crit.cuda()
    return crit


def cross_entropy_loss(scores, targets, criterion):
    scoresc = scores.view(-1, scores.size(2))   # (targets.size(0)*batch, voca_length_tgt)
    # yang didn't change targets into targets.view(-1) targets = targets.view(-1)
    loss = criterion(scoresc, targets.contiguous().view(-1))

    pred = torch.max(scores, 2)[1]
    num_correct = pred.eq(targets).masked_select(targets.ne(dict.PAD)).sum().item()
    num_total = targets.ne(dict.PAD).sum().item()
    return loss, num_total, num_correct
