"""
  rnn based predictor
  patient attribute is considered
  activity attribute is not considered
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

class WeightedRNNPtAttr:

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

    def model_architecture(self):
      # input layer    
      main_input = Input(shape=(self.maxlen,)) 
      
      # embedding layer
      num_words = len(self.word_index) + 2
      if self.embed is 'ACT2VEC_EMBED':
        (_, X, _), _, _, _, _= load_data(train_path = self.vec_path, dic_path = self.dic_path, config = self.config, valid_portion= 0, shuffle=False)

        embed_matrix = act2vec(X, self.word_index, self.embedding_dim, self.act2vec_win)
        embedding_output = Embedding(num_words, self.embedding_dim, weights=[embed_matrix], 
                                    input_length=self.maxlen, mask_zero=True)(main_input)
      elif self.embed is 'EMBED':
        embedding_output = Embedding(num_words, self.embedding_dim,
                                     input_length=self.maxlen, mask_zero=True)(main_input)
      elif self.embed is 'HOT':
        self.X_train = vec(self.X_train, self.word_index, 1, 0, self.maxlen)
        self.X_val = vec(self.X_val, self.word_index, 1, 0, self.maxlen) 
        self.X_test = vec(self.X_test, self.word_index, 1, 0, self.maxlen) 
        main_input = Input(shape=(self.maxlen, num_words)) 
        embedding_output = main_input

      # context attributes
      lengths = [len(v) for v in self.attr_dict.values()]
      pt_attr = Input(shape=(self.maxlen, lengths[0])) 

      x = keras.layers.concatenate([embedding_output, pt_attr])

      if (self.model is 'LSTM'):
          rnn1 = LSTM(self.hidden_vector, dropout = self.drop_out, recurrent_dropout = self.drop_out, return_sequences = True)(x)
      elif (self.model is 'GRU'):
          rnn1 = GRU(self.hidden_vector, dropout = self.drop_out, recurrent_dropout = self.drop_out, return_sequences = True)(x)
                           
      output = TimeDistributed(Dense(num_words, activation='softmax'))(rnn1)

      model = Model(inputs=[main_input, pt_attr], outputs=output)
      return model

    def model_architecture_dense(self):
        # input layer    
        main_input = Input(shape=(self.maxlen,)) 
        
        # embedding layer
        num_words = len(self.word_index) + 2
        if self.embed is 'ACT2VEC_EMBED':
          (_, X, _), _, _, _, _= load_data(train_path = self.vec_path, dic_path = self.dic_path, config = self.config, valid_portion= 0, shuffle=False)

          embed_matrix = act2vec(X, self.word_index, self.embedding_dim, self.act2vec_win)
          embedding_output = Embedding(num_words, self.embedding_dim, weights=[embed_matrix], 
                                      input_length=self.maxlen, mask_zero=True)(main_input)
        elif self.embed is 'EMBED':
          embedding_output = Embedding(num_words, self.embedding_dim, input_length=self.maxlen, mask_zero=True)(main_input)
        elif self.embed is 'HOT':
          self.X_train = vec(self.X_train, self.word_index, 1, 0, self.maxlen)
          self.X_val = vec(self.X_val, self.word_index, 1, 0, self.maxlen) 
          self.X_test = vec(self.X_test, self.word_index, 1, 0, self.maxlen) 
          main_input = Input(shape=(self.maxlen, num_words)) 
          embedding_output = main_input

        # context attributes
        lengths = [len(v) for v in self.attr_dict.values()]
        pt_attr = Input(shape=(self.maxlen, lengths[0])) 
        demo_input_dense = TimeDistributed(Dense(self.dense_size, activation='tanh'))(pt_attr)
        # x = keras.layers.concatenate([embedding_output, pt_attr])
        x = embedding_output

        if (self.model is 'LSTM'):
            rnn1 = LSTM(self.hidden_vector, dropout = self.drop_out, recurrent_dropout = self.drop_out, return_sequences = True)(x)
        elif (self.model is 'GRU'):
            rnn1 = GRU(self.hidden_vector, dropout = self.drop_out, recurrent_dropout = self.drop_out, return_sequences = True)(x)

        rnn1 = keras.layers.concatenate([rnn1, demo_input_dense])                     
        output = TimeDistributed(Dense(num_words, activation='softmax'))(rnn1)

        model = Model(inputs=[main_input, pt_attr], outputs=output)

        return model

    def run(self):
      # build model
      if (self.dense):
          model = self.model_architecture_dense()
      else:
          model = self.model_architecture()
      
      adam = Adam(lr=0.01, decay=1e-6)
      model.compile(loss='categorical_crossentropy',
               optimizer=adam,
               metrics=[ 'accuracy'], sample_weight_mode="temporal")
               # metrics=[ 'accuracy', 'top_5_categorical_accuracy'], sample_weight_mode="temporal")

      model.summary()

      # early stop and model storage
      dir = os.getcwd() # get current working directory
      model_dir = "res_models/"
      if not os.path.exists(dir + '/' + model_dir):
          os.makedirs(dir + '/' + model_dir)
      self.best_weights_filepath = dir + "/" + model_dir + datetime.now().strftime('%Y-_%m_%d; %H_%M_%S;') + ' weighted_main.h5'
      earlyStopping= keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
      saveBestModel = keras.callbacks.ModelCheckpoint(self.best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

      # fit model and save model
      model.fit([self.X_train, self.train_attr], self.y_train, batch_size= self.batch_size, epochs = self.epochs,
          validation_data=([self.X_val, self.val_attr], self.y_val), class_weight=self.weights, callbacks=[earlyStopping, saveBestModel])
      model.load_weights(self.best_weights_filepath)
      y_proba = model.predict([self.X_test, self.test_attr])

      precision, recall, acc, top_k1, top_k2, top_k3 = get_all_scores(self.y_test, y_proba, self.config)
      score, *_  = model.evaluate([self.X_test, self.test_attr], self.y_test)
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

