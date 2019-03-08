import pickle
import os
import dict

class PreConfig(object):
    root_org = 'en_vi'
    root_save = 'data'
    # -----about dictionary------#
    filter_length = 0
    trunc_length = 0


opt = PreConfig()

def file2dict(file, dict, filter_length=0, trun_length=0, char=False):
    print("file:{} | abandon the sentence whose length longer than {}| cut the sentence \
    whose length longer than {} to {}".format(file, filter_length, trun_length, trun_length))
    with open(file, encoding='utf8') as f:
        for sent in f.readlines():
            tokens = list(sent.strip()) if char else sent.strip().split()
            if len(sent.strip().split()) > filter_length > 0:
                continue
            if trun_length > 0:
                tokens = tokens[:trun_length]
            for word in tokens:
                dict.add(word)
    print("[DONE] file: {}  's vocabulary size: {}".format(file, len(dict)))
    return dict


def makeindexseq(srcfile, tgtfile, dicts, savefile):
    print('making sentence to index sequence from [', srcfile, '] and [', tgtfile, ']')
    srcf = open(srcfile, encoding='utf8')
    tgtf = open(tgtfile, encoding='utf8')
    savesrcstrf = open(savefile+'src.str', encoding='utf8')
    savetgtstrf = open(savefile+'tgt.str', encoding='utf8')
    savesrcindf = open(savefile+'src.ind')
    savetgtindf = open(savefile+'tgt.ind')

    #convert

    srcf.close()
    tgtf.close()
    savesrcindf.close()
    savesrcstrf.close()
    savetgtindf.close()
    savetgtstrf.close()





def main():
    train_src = os.path.join(opt.root_org, 'train.src')
    train_tgt = os.path.join(opt.root_org, 'train.tgt')
    valid_src = os.path.join(opt.root_org, 'valid.src')
    valid_tgt = os.path.join(opt.root_org, 'valid.tgt')
    test_src = os.path.join(opt.root_org, 'test.src')
    test_tgt = os.path.join(opt.root_org, 'test.tgt')

    # make dictionary, (use the train set)
    dicts = {}
    print("Building source vocabulary...")
    dicts['src'] = dict.Dict([dict.PAD_WORD, dict.UNK_WORD, dict.BOS_WORD, dict.EOS_WORD])
    dicts['src'] = file2dict(train_src, dicts['src'], opt.filter_length, opt.trunc_length)
    print("Building target vocabulary...")
    dicts['tgt'] = dict.Dict([dict.PAD_WORD, dict.UNK_WORD, dict.BOS_WORD, dict.EOS_WORD])
    dicts['tgt'] = file2dict(train_tgt, dicts['tgt'], opt.filter_length, opt.trunc_length)
    print("Saving dictionary...")
    dicts['src'].writefile(os.path.join(opt.root_save, 'src.dict'))
    dicts['tgt'].writefile(os.path.join(opt.root_save, 'tgt.dict'))

    # project sentence with word   to   sentence with index
    makeindexseq(train_src, train_tgt, dicts, os.path.join(opt.root_save, 'train'))
    makeindexseq(valid_src, valid_tgt, dicts, os.path.join(opt.root_save, 'valid'))
    makeindexseq(test_src, test_tgt, dicts, os.path.join(opt.root_save, 'test'))



if __name__ == "__main__":
    main()
