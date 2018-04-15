from models.weighted_rnn import WeightedRNN
from models.weighted_rnn_ptattr import WeightedRNNPtAttr
from models.attention_rnn import AttentionRNN
from models.attention_rnn_ptattr import AttentionRNNPtAttr

import numpy as np
import sys



seed = 613213
# seed = 0
np.random.seed(seed)

config_1 = {
    'hidden_vector' : 64,
    'drop_out'      : 0.5,
    'embedding_dim' : 17,
    'maxlen'        : 20,
    'batch_size'    : 20,
    'epochs'        : 10,
    'act2vec_win'   : 5,
    'dense'         : False,     # decide whether compress the patient attributes
    'dense_size'    : 3,         # decide the compressed patient attribute set size
    'attention'     : 'win',     # 'win': window size; 'pre': previous states; 'all': all states;
    'dataset'       : 'intubation',
    'attention_win' : 5,
    'random_seed'   : seed,
    'top_k1'        : 3,
    'top_k2'        : 5,
    'top_k3'        : 7,
}

train_path = 'data_intubation/s_act_withId.pkl'
dic_path = 'data_intubation/s_dict_withId.pkl'
# vec_path = 'data_intubation/s2000_log.pkl'
vec_path = 'data_intubation/s_act_withId.pkl'
attr_path = 'data_intubation/s_attrs_dict.pkl'


# train_path = 'data_synthetic_intubation/5000_log.pkl'
# dic_path = 'data_synthetic_intubation/dict_withIid.pkl'
# vec_path = 'data_synthetic_intubation/5000_log.pkl'
# attr_path = 'data_synthetic_intubation/attrs_dict.pkl'

if __name__ == '__main__':
    config_1['random_seed'] = int(sys.argv[2])
    np.random.seed(config_1['random_seed'])

    # Exp1: one-hot LSTM
    if (sys.argv[1] == '1'):
        weighted_lstm_hot = WeightedRNN(model = 'LSTM', embed='HOT', train_path = train_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # weighted_lstm_hot.cross_validation()
        weighted_lstm_hot.run()
    
    # Exp2: one-hot GRU
    if (sys.argv[1] == '2'):
        weighted_gru_hot = WeightedRNN(model = 'GRU', embed = 'HOT', train_path = train_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        weighted_gru_hot.run()

    # Exp3: LSTM + embedding
    if (sys.argv[1] == '3'):
        weighted_lstm_embed = WeightedRNN(model = 'LSTM',  embed = 'EMBED', train_path = train_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # weighted_lstm_embed.cross_validation()
        weighted_lstm_embed.run()

    # Exp4: GRU + embedding
    if (sys.argv[1] == '4'):
        weighted_gru_embed = WeightedRNN(model = 'GRU', embed = 'EMBED', train_path = train_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # weighted_gru_embed.cross_validation()
        weighted_gru_embed.run()

    # Exp5: LSTM + Act2vec + embedding (win_size = 5)
    if (sys.argv[1] == '5'):
        weighted_lstm_act2vec_embed = WeightedRNN(model = 'LSTM',  embed = 'ACT2VEC_EMBED', train_path = train_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        weighted_lstm_act2vec_embed.run()

    # Exp6: GRU + Act2vec + embedding (win_size = 5)
    if (sys.argv[1] == '6'):
        weighted_gru_act2vec_embed = WeightedRNN(model = 'GRU',  embed = 'ACT2VEC_EMBED', train_path = train_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        weighted_gru_act2vec_embed.run()

    # Exp7: LSTM + embedding + PatientAttributes
    if (sys.argv[1] == '7'):
        weighted_lstm_ptattr_embed = WeightedRNNPtAttr(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        weighted_lstm_ptattr_embed.run()

    # Exp8: GRU + embedding + PatientAttributes
    if (sys.argv[1] == '8'):
        weighted_gru_ptattr_embed = WeightedRNNPtAttr(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        weighted_gru_ptattr_embed.run()

    # Exp9: LSTM + embedding + Act2vec + PatientAttributes
    if (sys.argv[1] == '9'):
        weighted_lstm_ptattr_act2vec_embed = WeightedRNNPtAttr(model = 'LSTM',  embed = 'ACT2VEC_EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        weighted_lstm_ptattr_act2vec_embed.run()

    # Exp10: GRU + embedding + Act2vec + PatientAttributes
    if (sys.argv[1] == '10'):
        weighted_gru_ptattr_act2vec_embed = WeightedRNNPtAttr(model = 'GRU',  embed = 'ACT2VEC_EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        weighted_gru_ptattr_act2vec_embed.run()

    # Exp11: LSTM + embedding + PatientAttributes(dense) 
    if (sys.argv[1] == '11'):
        config_1['dense'] = True
        weighted_lstm_ptattr_embed = WeightedRNNPtAttr(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        weighted_lstm_ptattr_embed.run()

    # Exp12: GRU + embedding + PatientAttributes(dense) 
    if (sys.argv[1] == '12'):
        config_1['dense'] = True
        weighted_gru_ptattr_embed = WeightedRNNPtAttr(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
       # weighted_gru_ptattr_embed.cross_validation() 
        weighted_gru_ptattr_embed.run()   

    # Exp13: LSTM + embedding + Act2vec + PatientAttributes(dense) 
    if (sys.argv[1] == '13'):
        config_1['dense'] = True
        weighted_lstm_ptattr_embed = WeightedRNNPtAttr(model = 'LSTM',  embed = 'ACT2VEC_EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        weighted_lstm_ptattr_embed.run()

    # Exp14: GRU + embedding + Act2vec + PatientAttributes(dense) 
    if (sys.argv[1] == '14'):
        config_1['dense'] = True
        weighted_gru_ptattr_embed = WeightedRNNPtAttr(model = 'GRU',  embed = 'ACT2VEC_EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        weighted_gru_ptattr_embed.run() 

######################################
    # Exp46: PaPer + PATTR + GRU + embedding + attention(win)
    if (sys.argv[1] == '15'):
        config_1['dense'] = True
        # config_1['attention'] = 'noname_share'
        # attention_rnn_ptattr = AttentionRNNPtAttr(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # attention_rnn_ptattr.run()

        # config_1['attention'] = 'noname_no_share'
        # attention_rnn_ptattr = AttentionRNNPtAttr(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # attention_rnn_ptattr.run()

        config_1['attention'] = 'general'
        attention_rnn_ptattr = AttentionRNNPtAttr(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

        # config_1['attention'] = 'concat'
        # attention_rnn_ptattr = AttentionRNNPtAttr(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # attention_rnn_ptattr.run()

        # config_1['attention'] = 'ACL_simple'
        # attention_rnn_ptattr = AttentionRNNPtAttr(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # attention_rnn_ptattr.run()

        # config_1['attention'] = 'ACL_with_W'
        # attention_rnn_ptattr = AttentionRNNPtAttr(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # attention_rnn_ptattr.run()


    # Exp47: PaPer + LSTM + embedding + attention(win)
    if (sys.argv[1] == '16'):
        # config_1['attention'] = 'noname_share'
        # attention_rnn_ptattr = AttentionRNN(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # attention_rnn_ptattr.run()

        # config_1['attention'] = 'noname_no_share'
        # attention_rnn_ptattr = AttentionRNN(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # attention_rnn_ptattr.run()

        config_1['attention'] = 'general'
        attention_rnn_ptattr = AttentionRNN(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

        # config_1['attention'] = 'concat'
        # attention_rnn_ptattr = AttentionRNN(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # attention_rnn_ptattr.run()

        # config_1['attention'] = 'ACL_simple'
        # attention_rnn_ptattr = AttentionRNN(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # attention_rnn_ptattr.run()

        # config_1['attention'] = 'ACL_with_W'
        # attention_rnn_ptattr = AttentionRNN(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        # attention_rnn_ptattr.run()

    # Exp48: PaPer + LSTM + embedding + attention(win)
    if (sys.argv[1] == '17'):
        config_1['attention'] = 'noname_share'
        attention_rnn_ptattr = AttentionRNN(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

        config_1['attention'] = 'noname_no_share'
        attention_rnn_ptattr = AttentionRNN(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

        config_1['attention'] = 'general'
        attention_rnn_ptattr = AttentionRNN(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

        config_1['attention'] = 'concat'
        attention_rnn_ptattr = AttentionRNN(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

        config_1['attention'] = 'ACL_simple'
        attention_rnn_ptattr = AttentionRNN(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

        config_1['attention'] = 'ACL_with_W'
        attention_rnn_ptattr = AttentionRNN(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

    # Exp49: PaPer + LSTM + embedding + attention(win)
    if (sys.argv[1] == '18'):
        config_1['dense'] = True
        config_1['attention'] = 'noname_share'
        attention_rnn_ptattr = AttentionRNNPtAttr(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

        config_1['attention'] = 'noname_no_share'
        attention_rnn_ptattr = AttentionRNNPtAttr(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

        config_1['attention'] = 'general'
        attention_rnn_ptattr = AttentionRNNPtAttr(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

        config_1['attention'] = 'concat'
        attention_rnn_ptattr = AttentionRNNPtAttr(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

        config_1['attention'] = 'ACL_simple'
        attention_rnn_ptattr = AttentionRNNPtAttr(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

        config_1['attention'] = 'ACL_with_W'
        attention_rnn_ptattr = AttentionRNNPtAttr(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run()

    # Exp20: pre-trained model
    if (sys.argv[1] == '19'):
        config_1['attention'] = 'general'
        attention_rnn_ptattr = AttentionRNN(model = 'LSTM',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run_from_pretrained('res_models/pretrain_5000_intubation_LSTM.h5')

    # Exp20: pre-trained model
    if (sys.argv[1] == '20'):
        config_1['attention'] = 'general'
        attention_rnn_ptattr = AttentionRNN(model = 'GRU',  embed = 'EMBED', train_path = train_path, attr_path = attr_path, dic_path = dic_path, vec_path = vec_path, config = config_1)
        attention_rnn_ptattr.run_from_pretrained('res_models/pretrain_5000_intubation_LSTM.h5')
