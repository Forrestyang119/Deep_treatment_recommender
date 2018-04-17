"""
  rnn based predictor
  Neither patient attribute or activity attribute is not considered
"""
# Authors: Xin Dong
#          Sen Yang
#          Weiqing Ni


import keras
from keras.layers import Embedding, LSTM, GRU, TimeDistributed, Dense, Input, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam

import six.moves.cPickle as pickle
import gensim
from datetime import datetime
import os
import sys
from other.data_loader import * 
from other.tools import *
class WeightedRNN:

    def __init__(self, model, embed, train_path, dic_path, vec_path, config):
        """
        Parameters
        -----------
        embed : embeeding layer, three choices:
            'ACT2VEC_EMBED' Act2vec + embedding layer
            'EMBED' embedding layer
            'HOT' one-hot

        train_path: training data path

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

        (self.id_train, self.X_train, self.y_train), (self.id_val, self.X_val, self.y_val), (self.id_test, self.X_test, self.y_test), self.word_index, self.weights = load_data(train_path=train_path, dic_path = dic_path, config = config, shuffle=False)


    def model_architecture(self):
        # input layer    
        main_input = Input(shape=(self.maxlen,)) 
        
        # configure embedding layer
        num_words = len(self.word_index) + 2
        if self.embed == 'ACT2VEC_EMBED':
          (_, X, _), _, _, _, _= load_data(train_path = self.vec_path, dic_path = self.dic_path, config = self.config, valid_portion= 0, shuffle=False)

          embed_matrix = act2vec(X, self.word_index, self.embedding_dim, self.act2vec_win)
          embedding_output = Embedding(num_words, self.embedding_dim, weights=[embed_matrix], 
                                      input_length=self.maxlen, mask_zero=True)(main_input)
        elif self.embed == 'EMBED':
          embedding_output = Embedding(num_words, self.embedding_dim,
                                       input_length=self.maxlen, mask_zero=True)(main_input)
        elif self.embed == 'HOT':
          self.X_train = vec(self.X_train, self.word_index, 1, 0, self.maxlen)
          self.X_val = vec(self.X_val, self.word_index, 1, 0, self.maxlen) 
          self.X_test = vec(self.X_test, self.word_index, 1, 0, self.maxlen) 
          main_input = Input(shape=(self.maxlen, num_words)) 
          embedding_output = main_input

        # build the architecture
        if (self.model == 'LSTM'):
            rnn1 = LSTM(self.hidden_vector,
                         dropout = self.drop_out, recurrent_dropout = self.drop_out, return_sequences=True)(embedding_output)
        elif (self.model == 'GRU'):
            rnn1 = GRU(self.hidden_vector, dropout = self.drop_out, recurrent_dropout = self.drop_out, return_sequences = True)(embedding_output)
                             
        output = TimeDistributed(Dense(num_words, activation='softmax'))(rnn1)

        model = Model(inputs=main_input, outputs=output)

        return model


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
                label_pred.append(word_i[np.argmax(y_pred[i][j])])

            with open(filename, 'a') as f:
                wr = csv.writer(f, delimiter=',')
                wr.writerow([i])
                wr.writerow(['TRUE'] + label_true)
                wr.writerow(['PREDICT'] + label_pred)
        print()

    def run(self):
        # build model
        model = self.model_architecture()
        adam = Adam(lr=0.01, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'], sample_weight_mode="temporal")
                 # metrics=['precision', 'recall', 'accuracy', 'top_5_categorical_accuracy'], sample_weight_mode="temporal")

        print('Train...')
        print(np.shape(self.X_train))
        print(np.shape(self.X_val))
        

        dir = os.getcwd() # get current working directory
        model_dir = "res_models/"
        if not os.path.exists(dir + '/' + model_dir):
            os.makedirs(dir + '/' + model_dir)
        self.best_weights_filepath = dir + "/" + model_dir + datetime.now().strftime('%Y-_%m_%d; %H_%M_%S;') + ' weighted_main.h5'
        earlyStopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
        saveBestModel = keras.callbacks.ModelCheckpoint(self.best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        # fit model and save model
        model.fit(self.X_train, self.y_train, batch_size= self.batch_size, epochs = self.epochs,
            validation_data=(self.X_val, self.y_val), class_weight=self.weights, callbacks=[earlyStopping, saveBestModel])
        model.load_weights(self.best_weights_filepath)

        y_proba = model.predict(self.X_test)
        self.save_predict_result(self.y_test, y_proba, self.word_index)

        precision, recall, acc, top_k1, top_k2, top_k3 = get_all_scores(self.y_test, y_proba, self.config)
        score, *_  = model.evaluate(self.X_test, self.y_test)
        result_file = 'res_' + self.config['dataset'] + '.csv'
        write_result_to_file(result_file, [score, precision, recall, acc, top_k1, top_k2, top_k3], int(sys.argv[1]), self.config)
        print('Test score:', score)
        print('Precision:', precision)
        print('Recall:', recall)
        print('Test accuracy:', acc)
        print('top_k1 accuracy:', top_k1)
        print('top_k2 accuracy:', top_k2)
        print('top_k3 accuracy:', top_k3)
        print()

