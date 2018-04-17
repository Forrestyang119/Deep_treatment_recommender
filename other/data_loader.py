"""
    Loading data and preprocess
"""
import six.moves.cPickle as pickle
import numpy as np
from keras.preprocessing import sequence
import gensim
 
class Data:
    avg_length = 0
    max_length = 0

    def get_word_index(self, path="dict2.pkl"):
        path =  path
        if path.endswith(".gz"): 
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')
        word_dict = pickle.load(f)

        return word_dict

    def get_maxlen(self):
        return max_length

    def get_avglen(self):
        return avg_length

    def load_data(self, path="act.pkl", n_words=10000000, valid_portion=0, maxlen=None,
                  shuffle=False):

        # Load the dataset

        path = path
        if path.endswith(".gz"):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')

        train_set = pickle.load(f)
        f.close()

        print(maxlen)
        if maxlen: # if None???
            new_train_ids = []
            new_train_set_x = []
            new_train_set_y = []
            for id, x, y in zip(train_set[0], train_set[1], train_set[2]):
                if len(x) < maxlen:
                    new_train_ids.append(id)
                    new_train_set_x.append(x)
                    new_train_set_y.append(y)
            train_set = (new_train_ids, new_train_set_x, new_train_set_y)
            del new_train_ids, new_train_set_x, new_train_set_y

        # split training set into validation set
        train_ids, train_set_x, train_set_y = train_set
        # max length

        global max_length
        max_length = len(max(train_set_x, key=len))

        global avg_length
        avg_length = int(round(sum(map(len, train_set_x)) / len(train_set_x)))

        n_samples = len(train_set_x)
        if shuffle:
            sidx = np.random.permutation(n_samples)
        else:
            sidx = np.arange(n_samples)
        n_train = int(np.round(n_samples * (1. - valid_portion)))
        valid_set_ids = [train_ids[s] for s in sidx[n_train:]]
        valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
        valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
        train_set_ids = [train_ids[s] for s in sidx[:n_train]]
        train_set_x = [train_set_x[s] for s in sidx[:n_train]]
        train_set_y = [train_set_y[s] for s in sidx[:n_train]]

        train_set = (train_set_ids, train_set_x, train_set_y)
        valid_set = (valid_set_ids, valid_set_x, valid_set_y)

        def remove_unk(x): 
            return [[1 if w >= n_words else w for w in sen] for sen in x]

        valid_set_ids, valid_set_x, valid_set_y = valid_set
        train_set_ids, train_set_x, train_set_y = train_set

        train_set_x = remove_unk(train_set_x)
        valid_set_x = remove_unk(valid_set_x)

        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        def shuffle(train_ids, train_x, train_y):
            y_train = np.asarray(train_y)
            x_train = np.asarray(train_x)
            train_ids = np.asarray(train_ids)
            indices = np.arange(x_train.shape[0])
            
            # np.random.shuffle(indices) # shut down shuffle
            train_ids = train_ids[indices]
            x_train = x_train[indices]
            y_train = y_train[indices]
            return train_ids, x_train, y_train

        train = shuffle(train_set_ids, train_set_x, train_set_y)
        valid = (valid_set_ids, valid_set_x, valid_set_y)

        return train, valid


def load_data(train_path, dic_path, config, valid_portion=0.1, shuffle=False, k =1, s=1):
    data = Data()
    (id_train, X_train, y_train), (id_test, X_test, y_test) = data.load_data(path=train_path, 
            valid_portion=valid_portion, shuffle=shuffle)

    word_index = data.get_word_index(path=dic_path)
    print('Found %s unique tokens.' % len(word_index))
    print("max length %s" % data.get_maxlen())
    print("avg length %s" % data.get_avglen())

    from sklearn.utils import class_weight
    y =[]
    for yi in y_train:
        y += yi 
    y = np.asarray(y)
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y), y)
    print(class_weight)
    padding_w = np.asarray([1000, 1000])
    class_weight = np.concatenate((padding_w ,class_weight))
    class_weight = np.tile(class_weight,(config['maxlen'],1))
    print(class_weight.shape)
    print('Pad sequences (samples x time)')
    
    X_train = sequence.pad_sequences(X_train, maxlen=config['maxlen'])
    X_test = sequence.pad_sequences(X_test, maxlen=config['maxlen'])

    y_train = sequence.pad_sequences(y_train, maxlen=config['maxlen'])
    y_test = sequence.pad_sequences(y_test, maxlen=config['maxlen'])

    y_train = vec(y_train, word_index, k, s, config['maxlen'])
    y_test = vec(y_test, word_index, k, s, config['maxlen'])

    # split the data into a training set and a validation set
    indices = np.arange(X_train.shape[0])
    
    # np.random.shuffle(indices) # shut down shuffle
    trainIds = id_train[indices]
    traindata = X_train[indices]
    labels = y_train[indices]
    num_validation_samples = int(valid_portion * traindata.shape[0])

    if (num_validation_samples != 0):  # when valid_portion == 0, will have [0:-0]
        id_train = id_train[:-num_validation_samples]
        X_train = traindata[:-num_validation_samples]
        y_train = labels[:-num_validation_samples]
        id_val = id_train[-num_validation_samples:]
        X_val = traindata[-num_validation_samples:]
        y_val = labels[-num_validation_samples:]
    else:
        id_val = []
        X_val = []
        y_val = []

    # output_generator(y_test, word_index)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return (id_train, X_train, y_train), (id_val, X_val, y_val), (id_test, X_test, y_test), word_index, class_weight


def vec(org_y, word_index, k, s, maxlen):
    new_y = np.zeros((len(org_y), maxlen, len(word_index)+2))
    for i, y in enumerate(org_y):
        t0 = -1
        for t, a in enumerate(y):
            if not t0 == -1:
                for m in range(t-s, t-k-s, -1):
                    if m >= t0:
                        new_y[i, m, a] = 1
                        act = list(word_index.keys())[list(word_index.values()).index(a)]
                        # if (i < 1):
                        #     print(i, t, m, act)
                        
            elif a > 1: # a == 0 for paddings
                t0 = t
                new_y[i, t0, a] = 1

    return new_y

def load_attributes(path, id_train, id_val, id_test, maxlen):
    f = open(path, 'rb')
    attr_dict = pickle.load(f)
    f.close()

    train_attr = [] # this is list, needs to be converted to np array later
    val_attr = []
    test_attr = []
    for idx, id in enumerate(id_train):
        attr = attr_dict[id]
        train_attr.append([attr,] * maxlen)

    for idx, id in enumerate(id_val):
        attr = attr_dict[id]
        val_attr.append([attr, ] * maxlen)

    for idx, id in enumerate(id_test):
        attr = attr_dict[id]
        test_attr.append([attr, ] * maxlen)

    train_attr = np.asarray(train_attr)
    val_attr = np.asarray(val_attr)
    test_attr = np.asarray(test_attr)

    return attr_dict, train_attr, val_attr, test_attr


def act2vec(X_train, word_index, embedding_dim, win = 5):
    X_train = [[str(act) for act in trace if act != 0] for trace in X_train]
    model = gensim.models.Word2Vec(X_train, size=embedding_dim, alpha=0.1, window=win, min_count=0, workers=4, sg=1)
    # s =[model.wv.similarity('2', str(i)) for i in word_index.values()]
    embedding_matrix = np.zeros((len(word_index) + 2, embedding_dim))
    for index in word_index.values():
        try:
            embedding_vector = model.wv[str(index)]
        except KeyError:
            embedding_vector = None
            print(str(index) + ' is not in word vectors')
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[index] = embedding_vector
    return embedding_matrix


def output_generator(y_proba, word_index, y_test):
    activity_generator = []
    for ii in range(len(y_proba) - 1):
        all_activity = []
        for i in range(len(y_proba[ii]) - 1):
            now = y_proba[ii][i]
            next = y_proba[ii][i + 1]  # now and next is for padding detection
            if not (now == next).all():
                lst = np.zeros(len(y_proba[ii][i + 1]))
                ind = np.argpartition(y_proba[ii][i + 1], -5)[-5:]
                lst[ind] = 1
                for idx in range(len(y_proba[ii][i + 1]) - 1):
                    # print(ii,i+1,idx, y_proba[ii][i + 1][idx], lst[idx], y_test[ii][i+1][idx])
                    if lst[idx] == 1 or y_test[ii][i + 1][idx] == 0.2:
                        activity = list(word_index.keys())[list(word_index.values()).index(idx)]
                        print(ii, i + 1, idx, activity, lst[idx], y_test[ii][i + 1][idx])
                        #                 all_activity.append(lst)
                        # activity_generator.append(all_activity)
    return activity_generator
