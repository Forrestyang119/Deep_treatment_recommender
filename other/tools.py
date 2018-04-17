import numpy as np
import csv
from other.custom_layers import *
import matplotlib.pyplot as plt
import itertools
# return accuracy, precision, recall, top-k accuracy
def get_all_scores(y_true, y_pred, config):
  k1 = config['top_k1']
  k2 = config['top_k2']
  k3 = config['top_k3']
  # confusion matrix
  cm = [[0 for i in range(y_true.shape[2])] for i in range(y_true.shape[2])]
  top_k1_label = []
  top_k2_label = []
  top_k3_label = []
  true_label = []
  # calculate confusion matix
  for i in range(y_true.shape[0]):
  # i-th sequence
    for j in range(y_true.shape[1]):
      # j-th word
      if np.sum(y_true[i][j]) == 0:
        continue
      true_label.append(np.argmax(y_true[i][j]))
      top_k1_label.append(y_pred[i][j].argsort()[-k1:].tolist())
      top_k2_label.append(y_pred[i][j].argsort()[-k2:].tolist())
      top_k3_label.append(y_pred[i][j].argsort()[-k3:].tolist())
      idx_true, idx_pred = np.argmax(y_true[i][j]), np.argmax(y_pred[i][j])
      cm[idx_true][idx_pred] += 1
  cm = np.array(cm)
  accuracy = np.sum(cm.diagonal())/np.sum(cm)
  fn = [np.sum(cm[i]) - cm[i][i] for i in range(cm.shape[0])]
  fp = [np.sum(cm[:][i]) - cm[i][i] for i in range(cm.shape[0])]
  tp = [cm[i][i] for i in range(cm.shape[0])]
  precision, recall = 0, 0
  for i in range(cm.shape[0]):
    if tp[i] + fp[i] != 0:
      precision += tp[i] / (tp[i] + fp[i]) / cm.shape[0]
    if tp[i] + fn[i] != 0:
      recall += tp[i] / (tp[i] + fn[i]) / cm.shape[0]

  # precision = sum([tp[i] / (tp[i] + fp[i]) for i in range(cm.shape[0])]) / cm.shape[0]
  # recall = sum([tp[i] / (tp[i] + fn[i]) for i in range(cm.shape[0])]) / cm.shape[0]
  top_k1_correct = 0
  top_k2_correct = 0
  top_k3_correct = 0
  for v, k1, k2, k3, in list(zip(true_label, top_k1_label, top_k2_label, top_k3_label)):
    if v in k1:
      top_k1_correct += 1
    if v in k2:
      top_k2_correct += 1
    if v in k3:
      top_k3_correct += 1
  top_k1_acc = top_k1_correct / len(true_label)
  top_k2_acc = top_k2_correct / len(true_label)
  top_k3_acc = top_k3_correct / len(true_label)

  return  round(precision, 5), round(recall, 5), round(accuracy, 5), round(top_k1_acc, 5), round(top_k2_acc, 5), round(top_k3_acc, 5)

def write_result_to_file(file, results, exp_idx, config):
  exp_name = [
    "0  - invalid",
    "1  - one-hot LSTM",
    "2  - one-hot GRU",
    "3  - LSTM+EMB",
    "4  - GRU+EMB",
    "5  - LSTM+A2V+EMB(WIN=5)",
    "6  - GRU+A2V+EMB(WIN=5)",
    "7  - LSTM+EMB+PATTR",
    "8  - GRU+EMB+PATTR",
    "9  - LSTM+EMB+A2V+PATTR",
    "10 - GRU+EMB+A2V+PATTR",
    "11 - LSTM+EMB+PATTR(dense) ",
    "12 - GRU+EMB+PATTR(dense) ",
    "13 - LSTM+EMB+A2V+PATTR(dense) ",
    "14 - GRU+EMB+A2V+PATTR(dense) ",
    "15 - PaPer+LSTM+EMB+ATT",
    "16 - PaPer+PATTR+LSTM+EMB+ATT",
    "17 - PaPer+GRU+EMB+ATT",
    "18 - PaPer+PATTR+GRU+EMB+ATT",
    "19 - Pre-trained+LSTM+ATT(General)"
    "19 - Pre-trained+GRU+ATT(General)"
  ]

  with open(file, 'a') as f:
    if exp_idx > len(exp_name) - 1:
      print("name of ", exp_idx, "experiment: ")
      name = input()
    else:
      name = exp_name[exp_idx]
    wr = csv.writer(f, delimiter=',')
    wr.writerow(['Apr_10: ', name, config['hidden_vector'], config['embedding_dim'], config['batch_size'], config['dense'], config['dense_size'], config['attention'], config['attention_win'],  config['random_seed']] + results)

def attention_selector(config, rnn1):
  alphas = []

  if config['attention'] == 'general':
    attention_output, alphas = attention_context_gen_k(config, rnn1)

  if config['attention'] == 'concat':
    attention_output, alphas = attention_context_concat_k(config, rnn1)

  if config['attention'] == 'ACL_simple':
    attention_output, alphas = attention_layer_ACL_simple(config, rnn1)
  
  else:
    attention_output, alphas = attention_context_gen_k(config, rnn1)

  return attention_output, alphas

def plot_confusion_matrix(image_file, cm, classes,
                          title='Attention matrix',
                          cmap=plt.cm.Blues, w=5, acc=-1):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    title = 'acc: ' + str(acc * 100) + ' %'
    plt.title(title)
    plt.colorbar()
    y_tick_marks = np.arange(len(classes[0]))
    x_tick_marks = np.arange(len(classes[0]) + w)
    plt.xticks(x_tick_marks, ['NA' for i in range(w)] + classes[0], rotation=90)
    plt.yticks(y_tick_marks, classes[1])

    fmt = '.2f'
    thresh = cm.max() / 2.
    # plt.grid(ls='--', alpha=0.5)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i,j] != 0:
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=5)

    # plt.tight_layout()
    # plt.xlabel('True label')
    # plt.ylabel('Predicted label')
    # plt.show()  
    plt.savefig(image_file, dpi=300,bbox_inches='tight')
    plt.close()

def flatten_one_hot(y_true, y_pred):
  res = []
  for i in range(y_true.shape[0]):
    tmp = []
    for j in range(y_true.shape[1]):
      if np.sum(y_true[i][j]) == 0:
        tmp.append(0)
      else:
        tmp.append(np.argmax(y_pred[i][j]))
    res.append(tmp)
  return res

def flatten_one_hot_np(y):
  res = []
  for i in range(y.shape[0]):
    tmp = []
    for j in range(y.shape[1]):
      if np.sum(y[i][j]) == 0:
        tmp.append(0)
      else:
        tmp.append(np.argmax(y[i][j]))
    res.append(tmp)
  return np.array(res)