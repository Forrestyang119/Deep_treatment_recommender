"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

# dataset_path='D:/XinDong/Dropbox/dataset/'
# dataset_path = '/Users/Xin/Dropbox/dataset/'

# dataset_path = '122AustinDictionary_09.18.csv'
#dataset_path = 'random_Generated_sequence.csv'
import numpy
import six.moves.cPickle as pkl
import glob
import os
import pandas as pd

# dataset_path = os.getcwd() + '/s_output_2000.csv'
# dataset_path = 'output_5000_PIMA.csv'
dataset_path = 'Files_synthetic/rnn_input_1000.csv'

def build_dict():
    df = pd.read_csv(dataset_path)
    headers = list(df)
    data = []
    for name in headers:
        data.append(df[name].tolist())
    idlist =  list(set(data[0]))
    # dictoinary = list(set(data[1]))
    logs = {}
    for id in idlist:
        log = []
        for i in range(len(data[0])):
            if data[0][i]==id:
                log.append((data[1][i]))
        logs[id] = log

    print('Building dictionary..'),
    wordcount = dict()
    for words in logs.values():
        for w in words:
            w = w.lower()
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = list(wordcount.values())
    keys = list(wordcount.keys())

    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx + 2  # leave 0 and 1 (UNK)

    print(numpy.sum(counts), ' total words ', len(keys), ' unique words')

    return worddict, logs

def grab_data_labels(dictionary, sentences):

    seqs = [None] * len(sentences)
    caseIds = [None] * len(sentences)
    for idx, id in enumerate(sentences.keys()):
        words = sentences[id]
        seqs[idx] = [dictionary[w.lower()] if w.lower() in dictionary else 1 for w in words]
        caseIds[idx] = id

    labels = [None] * len(sentences)
    for i in range(len(seqs)):
        labels[i] = seqs[i][1:]

    return  caseIds, seqs, labels


def main(path = None):
    if path is None:
        dictionary, logs = build_dict()
        f = open('Files_synthetic/al_dict.pkl', 'wb')
        pkl.dump(dictionary, f, -1)
        f.close()
    else:
        f = open(path, 'rb')
        dictionary = pkl.load(f)
        _, logs = build_dict()
        f.close()
    caseIds, x, y = grab_data_labels(dictionary, logs)
    f = open('Files_synthetic/1000_log.pkl', 'wb')
    pkl.dump((caseIds,x, y), f, -1)
    f.close()


if __name__ == '__main__':
    # path = os.getcwd()  +  '/s_dict_withIid.pkl'
    # main(path = path)
    main()