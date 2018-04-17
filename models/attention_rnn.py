"""
  rnn based predictor
  patient attribute is considered
  activity attribute is not considered
"""
# Authors: Xin Dong
#          Sen Yang
#          Weiqing Ni

import pdb
import keras
from keras.layers import Embedding, LSTM, GRU, TimeDistributed, Dense, Input, Dropout
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers.core import *
from keras.layers import merge, Multiply, Add, Activation
from keras.optimizers import Adam
import numpy as np
import six.moves.cPickle as pickle
import gensim
from datetime import datetime
import os
from other.tools import *
import sys
from other.data_loader import * 
from keras import backend as K
from other.custom_layers import *
class AttentionRNN:

    def __init__(self, model, embed, train_path, attr_path, dic_path, vec_path, config):
        """
        Parameters
        -----------
        embed : embeeding layer, three choices:
            'ACT2VEC_EMBED' Act2vec + embedding layer
            'EMBED' embedding layer
            'HOT' one-hot

        train_path: training data path

        attr_path: patient attribute data path

        dic_path: dictionary of activity type

        vec_path: Act2vec input path

        Config: hyperparameters of model
        """
        self.model = model
        self.embed = embed
        self.vec_path = vec_path
        self.dic_path = dic_path
        self.config = config
        self.hidden_vector = config['hidden_vector']
        self.drop_out = config['drop_out']
        self.maxlen = config['maxlen']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.embedding_dim = config['embedding_dim']
        self.act2vec_win = config['act2vec_win']
        self.dense = config['dense']
        self.dense_size = config['dense_size']

        (self.id_train, self.X_train, self.y_train), (self.id_val, self.X_val, self.y_val), (self.id_test, self.X_test, self.y_test), self.word_index, self.weights = load_data(train_path=train_path, dic_path = dic_path, config = config, shuffle=False)
        
        self.attr_dict, self.train_attr, self.val_attr, self.test_attr = load_attributes(path = attr_path, id_train = self.id_train, id_val = self.id_val, id_test = self.id_test, maxlen = self.maxlen)

        self.num_words = len(self.word_index) + 2
        self.mask_train, self.mask_val, self.mask_test = [self.create_mask(a, self.num_words) for a in [self.X_train, self.X_val, self.X_test]]


    def create_mask(self, array, num_words):
        arr = np.copy(array)
        arr = arr.astype(float)
        arr = np.expand_dims(arr, axis = 2)
        arr = np.repeat(arr, num_words, axis = 2)
        arr[arr > 0] = 1
        arr[arr == 0] = 1e-8

        return arr


    def embedding_layer(self, main_input):
        # Pre-trained act-vec layer
        if self.embed is 'ACT2VEC_EMBED':
            (_, X, _), _, _, _, _= load_data(train_path = self.vec_path, dic_path = self.dic_path, config = self.config, valid_portion= 0, shuffle=False)
            embed_matrix = act2vec(X, self.word_index, self.embedding_dim, self.act2vec_win)
            embedding_output = Embedding(self.num_words, self.embedding_dim, weights=[embed_matrix], 
                                          input_length=self.maxlen, mask_zero=True)(main_input)
        
        # Word Embedding layer
        elif self.embed is 'EMBED':
            embedding_output = Embedding(self.num_words, self.embedding_dim,
                                   input_length=self.maxlen, mask_zero=True)(main_input)
        
        # One hot layer
        elif self.embed is 'HOT':
            self.X_train = vec(self.X_train, self.word_index, 1, 0, self.maxlen)
            self.X_val = vec(self.X_val, self.word_index, 1, 0, self.maxlen) 
            self.X_test = vec(self.X_test, self.word_index, 1, 0, self.maxlen) 
            main_input = Input(shape=(self.maxlen, num_words)) 
            embedding_output = main_input

        return embedding_output



    def model_architecture(self):
        num_words = len(self.word_index) + 2

        # input layer    
        mask_input = Input(shape = (self.maxlen, num_words)) # input for mask
        main_input = Input(shape = (self.maxlen,))              # input should be a tensor
        
        # embedding layer
        embedding_output = self.embedding_layer(main_input)

        # rnn layer
        if (self.model is 'LSTM'):
            rnn1 = LSTM(self.hidden_vector, dropout = self.drop_out, recurrent_dropout = self.drop_out, return_sequences = True)(embedding_output)
        elif (self.model is 'GRU'):
            rnn1 = GRU(self.hidden_vector, dropout = self.drop_out, recurrent_dropout = self.drop_out, return_sequences = True)(embedding_output)
        
        rnn1 = Lambda(lambda x : x, output_shape = lambda s: s)(rnn1)

        # attention layer
        attention_output, self.alphas = attention_selector(self.config, rnn1)       

        # output layer
        output = TimeDistributed(Dense(num_words, activation='softmax'))(attention_output)


        model = Model(inputs=[main_input, mask_input], outputs=output)

        return model


    def model_evaluation(self, model):

        # Get y_predict
        y_proba = model.predict([self.X_test, self.mask_test])
        self.save_predict_result(self.y_test, y_proba, self.word_index)


        precision, recall, acc, top_k1, top_k2, top_k3 = get_all_scores(self.y_test, y_proba, self.config)
        score, *_  = model.evaluate([self.X_test, self.mask_test], self.y_test)
        print('Test_acc = ', acc * 100, '%')
        print('Test score:', score)
        print('Precision:', precision)
        print('Recall:', recall)
        print('Test accuracy:', acc)
        print('top_k1 accuracy:', top_k1)
        print('top_k2 accuracy:', top_k2)
        print('top_k3 accuracy:', top_k3)
        result_file = 'res_' + self.config['dataset'] + '.csv'
        write_result_to_file(result_file, [score, precision, recall, acc, top_k1, top_k2, top_k3], int(sys.argv[1]), self.config)
        print()


    def save_predict_result(self, y_true, y_pred, word_index):   
        dir = os.getcwd()
        if not os.path.exists(dir + '/res_pred/'):
            os.makedirs(dir + '/res_pred/')

        filename = dir + '/res_pred/' + datetime.now().strftime('%Y-_%m_%d; %H_%M_%S;') + '.csv'
        # reverse dict
        word_i = dict([(v, k) for k, v in word_index.items()])
        import csv
        for i in range(y_true.shape[0]):
            # i-th sequence
            # transfer one-hot/softmax to label
            label_true, label_pred = [], []
            for j in range(y_true.shape[1]):
                if np.sum(y_true[i][j]) == 0:
                    continue
                label_true.append(word_i[np.argmax(y_true[i][j])])
                if np.argmax(y_pred[i][j]) in [0,1]:
                    label_pred.append('[PAD]')
                else:
                    label_pred.append(word_i[np.argmax(y_pred[i][j])])

            with open(filename, 'a') as f:
                wr = csv.writer(f, delimiter=',')
                wr.writerow([i])
                wr.writerow(['TRUE'] + label_true)
                wr.writerow(['PREDICT'] + label_pred)
        print()

    def run(self):
        # build model
        if (self.dense):
            model = self.model_architecture_dense()
        else:
            model = self.model_architecture()
        
        adam = Adam(lr=0.01, decay=1e-6)
        # compile model
        model.compile(loss='categorical_crossentropy',
                 optimizer=adam,
                 metrics=['accuracy'], sample_weight_mode="temporal")
                 # metrics=['accuracy', 'top_3_categorical_accuracy'], sample_weight_mode="temporal")

        model.summary()

        # early stop and model storage
        # loss or acc:https://stackoverflow.com/questions/37141636/should-i-use-loss-or-accuracy-as-the-early-stopping-metric
        dir = os.getcwd() # get current working directory
        model_dir = "res_models/"
        if not os.path.exists(dir + '/' + model_dir):
            os.makedirs(dir + '/' + model_dir)
        self.best_weights_filepath = dir + "/" + model_dir + datetime.now().strftime('%Y-_%m_%d; %H_%M_%S;') + ' weighted_main.h5'

        earlyStopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
        saveBestModel = keras.callbacks.ModelCheckpoint(self.best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        # fit model and save model
        model.fit([self.X_train, self.mask_train], self.y_train, batch_size= self.batch_size, epochs = self.epochs,
            validation_data=([self.X_val, self.mask_val], self.y_val), class_weight=self.weights, callbacks=[earlyStopping, saveBestModel])
        model.load_weights(self.best_weights_filepath)
       
        # evaluate model
        self.model_evaluation(model)


    def run_from_pretrained(self, model_path):
        # build model
        if (self.dense):
            model = self.model_architecture_dense()
        else:
            model = self.model_architecture()
        
        adam = Adam(lr=0.01, decay=1e-6)
        # adam = 'adam'
        # compile model
        model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'], sample_weight_mode="temporal")
                 # metrics=['accuracy', 'top_3_categorical_accuracy'], sample_weight_mode="temporal")
        model.load_weights(model_path)

        # early stop and model storage
        # loss or acc:https://stackoverflow.com/questions/37141636/should-i-use-loss-or-accuracy-as-the-early-stopping-metric
        dir = os.getcwd() # get current working directory
        self.best_weights_filepath = dir + "/res_models/" + datetime.now().strftime('%Y-_%m_%d; %H_%M_%S;') + ' weighted_main.h5'
        earlyStopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
        saveBestModel = keras.callbacks.ModelCheckpoint(self.best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        # fit model and save model
        model.fit([self.X_train, self.mask_train], self.y_train, batch_size= self.batch_size, epochs = self.epochs,
            validation_data=([self.X_val, self.mask_val], self.y_val), class_weight=self.weights, callbacks=[earlyStopping, saveBestModel])
        model.load_weights(self.best_weights_filepath)

        # evaluate model
        self.model_evaluation(model)
        # get attention output
        if not self.config['attention'] in ['general', 'concat', 'ACL_simple']:
            return 
        new_model = Model(inputs=model.input, outputs=self.alphas)
        result = new_model.predict([self.X_train, self.mask_train])

        # get prediction
        y_predict = model.predict([self.X_train, self.mask_train])
        y_predict = flatten_one_hot(self.y_train,y_predict)
        self.plot_attention_matrix(self.X_train, result, y_predict)
        print('')

    def plot_attention_matrix(self, data, matrix, label_pred):
        matrix = np.concatenate(matrix, axis = 1)

        time_step = matrix.shape[1]
        win_size = matrix.shape[2]
        # reverse word_index dictionary
        word_dict = {}
        for k,v in self.word_index.items():
            if 'pre-oxygenation' in k:
                k = k.replace('pre-oxygenation', 'pre-oxy.')
            if 'post-oxygenation' in k:
                k = k.replace('post-oxygenation', 'post-oxy.')
            if 'verbalized' in k:
                k = k.replace('verbalized', 'verb.')
            if 'auscultation' in k:
                k = k.replace('auscultation', 'auscul.')
            word_dict[v] = k
        image_dir = 'attention_visualization/'
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        for i in range(matrix.shape[0]):
            acts_true = []
            acts_pred = []
            new_matrix = np.zeros((time_step, time_step + win_size))
            correct_num = 0
            # i - th sequence
            for j in range(time_step):
                # j-th time step
                new_matrix[j, j:j+win_size] = matrix[i,j,:]
                if data[i][j] == 0:
                    acts_true.append("[PAD]")
                else:
                    acts_true.append(word_dict[data[i][j]])
                if label_pred[i][j] in [0,1]:
                    acts_pred.append("[PAD]")
                else:
                    acts_pred.append(word_dict[label_pred[i][j]])
                if j < time_step - 1 and data[i][j+1] == label_pred[i][j]:
                    correct_num += 1
            # calculate accuracy
            acc = correct_num/(time_step-1)
            image_file = image_dir + str(i) + '.png'
            plot_confusion_matrix(image_file,new_matrix, classes=[acts_true, acts_pred],w=win_size, acc=acc)

    def plot_attention_matrix_all(self, data, matrix, label_pred):
        # attention_weights = np.concatenate(matrix, axis = 1)
        import pdb

        num_seqs = matrix[0].shape[0]
        time_step = self.maxlen
        # win_size = matrix.shape[2]
        # reverse word_index dictionary
        word_dict = {}
        for k,v in self.word_index.items():
            if 'pre-oxygenation' in k:
                k = k.replace('pre-oxygenation', 'pre-oxy.')
            if 'post-oxygenation' in k:
                k = k.replace('post-oxygenation', 'post-oxy.')
            if 'verbalized' in k:
                k = k.replace('verbalized', 'verb.')
            if 'auscultation' in k:
                k = k.replace('auscultation', 'auscul.')
            word_dict[v] = k
        image_dir = 'attention_visualization/'
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        for i in range(num_seqs):
            # i-th sequence
            new_matrix = []
            acts_true = []
            acts_pred = []
            correct_num = 0
            for j in range(len(matrix)):
                # j-th timestep
                alpha = (matrix[j].tolist())[i][0]
                alpha += [0 for i in range(time_step - len(alpha)+1)]
                new_matrix.append(alpha)

            new_matrix = np.array(new_matrix)

            # get labels
            for j in range(time_step):
                # j-th time step
                if data[i][j] == 0:
                    acts_true.append("[PAD]")
                else:
                    acts_true.append(word_dict[data[i][j]])
                if label_pred[i][j] == 0:
                    acts_pred.append("[PAD]")
                else:
                    acts_pred.append(word_dict[label_pred[i][j]])
                if j < time_step - 1 and data[i][j+1] == label_pred[i][j]:
                    correct_num += 1            
            acc = correct_num/(time_step - 1)
            image_file = image_dir + str(i) + '.png'
            # plt.savefig(image_file)
            plot_confusion_matrix(image_file,new_matrix, classes=[acts_true, acts_pred],w=1, acc=acc)
