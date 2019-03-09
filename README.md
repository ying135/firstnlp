# This is my first nlp code, seq2seq code to translate

## preprocess.py
### data (dictionary) structrue

key | value (also dictionary)
---- | ---
'train' | trains
'valid' |  valids
'test' |  tsts
'dicts' |  dicts

#### trains valids tests structure

key | value (path to file)
---- | ---
'srcindf' | *.src.ind
'tgtindf' | *.tgt.ind
'srcstrf' | *.src.str
'tgtstrf' | *.tgt.str
'length'  | count