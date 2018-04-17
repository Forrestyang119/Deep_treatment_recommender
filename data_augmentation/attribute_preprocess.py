import numpy
import six.moves.cPickle as pkl
import glob
import os
import pandas as pd

# dataset_path = '122AustinDictionary_09.18.csv'
dataset_path = 'Files_synthetic/rnn_input_1000.csv'

def build_dict():
    df = pd.read_csv(dataset_path)
    headers = list(df)

    # import sys
    # sys.exit()

    data = []
    for name in headers:
        data.append(df[name].tolist())
    idlist =  list(set(data[0]))
    # dictoinary = list(set(data[1]))
    attrs = dict()
    for i in range(len(idlist)):
        attr = []
        for j in range(2,len(data)):
            attr.append(data[j][i])
        attrs[idlist[i]] = attr
    print('attribute dict built')
    print(attrs)
    return attrs


def main():
    dictionary = build_dict()

    f = open('Files_synthetic/attrs_dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()

if __name__ == '__main__':
    main()

