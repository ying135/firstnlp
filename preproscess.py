import pickle
import os
import dict


class PreConfig(object):
    root_org = 'en_vi'
    root_save = 'data'
    # -----about dictionary------#
    filter_length_src = 0   # abandon the sentence whose length longer than it
    filter_length_tgt = 0
    trunc_length_src = 0    # cut the sentence whose length longer than it
    trunc_length_tgt = 0
    src_char = False
    tgt_char = False
    lower = True
    # ---
    prep_iter = 2000


opt = PreConfig()

def file2dict(file, dict, filter_length=0, trun_length=0, char=False):
    print("file:{} | filter_length: {} | trun_length: {}".format(file, filter_length, trun_length))
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


# this function for many2many
def makeindexseq(srcfile, tgtfile, dicts, savefile):
    print('making sentence to index sequence from file: ', srcfile, ' and ', tgtfile)
    srcf = open(srcfile, encoding='utf8')
    tgtf = open(tgtfile, encoding='utf8')
    savesrcstrf = open(savefile+'src.str', 'w', encoding='utf8')
    savetgtstrf = open(savefile+'tgt.str', 'w', encoding='utf8')
    savesrcindf = open(savefile+'src.ind', 'w')
    savetgtindf = open(savefile+'tgt.ind', 'w')

    #convert
    count = 0 # the number of sentences whose length meet the requirements
    oricount = 0
    while True:
        sline = srcf.readline()
        tline = tgtf.readline()

        # i think that determine whether it's end of file or not in this way is not good
        if sline == "" and tline == "":
            break
        if sline =="" or tline == "":
            print("[WARNING] the length of source and target file are not equal")
            break

        oricount += 1
        sline = sline.strip()
        tline = tline.strip()
        if sline == "" or tline == "":
            print("[WARNING] ignoring an empty line: "+str(oricount))
            continue
        if opt.lower:
            sline.lower()
            tline.lower()
        srcwords = list(sline) if opt.src_char else sline.split()
        tgtwords = list(tline) if opt.tgt_char else tline.split()
        if len(sline.split()) > opt.filter_length_src > 0 or len(tline.split()) > opt.filter_length_tgt > 0:
            continue
        if opt.trunc_length_src > 0:
            srcwords = srcwords[:opt.trunc_length_src]
        if opt.trunc_length_tgt > 0:
            tgtwords = tgtwords[:opt.trunc_length_tgt]

        srcinds = dicts['src'].words2idx(srcwords, dict.UNK_WORD)
        tgtinds = dicts['tgt'].words2idx(tgtwords, dict.UNK_WORD, dict.BOS_WORD, dict.EOS_WORD)
        savesrcindf.write(" ".join(list(map(str, srcinds))) + '\n')
        savetgtindf.write(" ".join(list(map(str, tgtinds))) + '\n')
        a = "" if opt.src_char else " "
        b = "" if opt.tgt_char else " "
        savesrcstrf.write(a.join(srcwords)+'\n')
        savetgtstrf.write(b.join(tgtwords)+'\n')
        count += 1
        if count % opt.prep_iter == 0:
            print("{} sentences are done".format(count))

    srcf.close()
    tgtf.close()
    savesrcindf.close()
    savesrcstrf.close()
    savetgtindf.close()
    savetgtstrf.close()
    print('[DONE]')
    return {'srcindf':savefile+'src.ind', 'tgtindf':savefile+'tgt.ind',\
            'srcstrf': savefile+'src.str', 'tgtstrf':savefile+'tgt.str', 'length':count}


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
    dicts['src'] = file2dict(train_src, dicts['src'], opt.filter_length_src, opt.trunc_length_src)
    print("Building target vocabulary...")
    dicts['tgt'] = dict.Dict([dict.PAD_WORD, dict.UNK_WORD, dict.BOS_WORD, dict.EOS_WORD])
    dicts['tgt'] = file2dict(train_tgt, dicts['tgt'], opt.filter_length_tgt, opt.trunc_length_tgt)
    print("Saving dictionary...")
    dicts['src'].writefile(os.path.join(opt.root_save, 'src.dict'))
    dicts['tgt'].writefile(os.path.join(opt.root_save, 'tgt.dict'))

    # project sentence with word   to   sentence with index
    trains = makeindexseq(train_src, train_tgt, dicts, os.path.join(opt.root_save, 'train.'))
    valids = makeindexseq(valid_src, valid_tgt, dicts, os.path.join(opt.root_save, 'valid.'))
    tests = makeindexseq(test_src, test_tgt, dicts, os.path.join(opt.root_save, 'test.'))

    data = {'train': trains, 'valid': valids, 'test': tests, 'dicts':dicts}
    output = open(os.path.join(opt.root_save, 'data.pkl'), 'wb')
    pickle.dump(data, output)
    output.close()


if __name__ == "__main__":
    main()
