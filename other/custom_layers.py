import keras
from keras.layers import Embedding, LSTM, GRU, TimeDistributed, Dense, Input, Dropout
from keras.models import Model, Sequential
from keras.layers.core import *
from keras.layers import merge, Multiply, Add, Activation, dot, concatenate
from keras import backend as K


def attention_layer(config, rnn1):
    input_dim = int(rnn1.shape[2])
    attention_tensors = []
    # attention_output = rnn1[: 1, :]
    BK_WIN = config['attention_win']
    for i in range(1, config['maxlen'] + 1, 1):
        dense_1 = Dense(BK_WIN)
        if i < BK_WIN:
            rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
            attr = Permute((2,1))(rnn_slice)
            a_1 = Reshape((input_dim, i))(attr)
            a_2 = Dense(i, activation='softmax')(a_1)
            a_probs = Permute((2,1))(a_2)
            attention_mul = merge([a_probs, rnn_slice], mode='mul')
            # attention_mul = Lambda(lambda x: x[:, i-1:i, :])(attention_mul)
        else:
            rnn_slice = Lambda(lambda x : x[:, i - BK_WIN:i, :])(rnn1)
            attr = Permute((2,1))(rnn_slice)
            a_1 = Reshape((input_dim, BK_WIN))(attr)
            a_2 = Dense(BK_WIN, activation='softmax')(a_1)
            a_probs = Permute((2,1))(a_2)
            attention_mul = merge([a_probs, rnn_slice], mode='mul')
            # attention_mul = Lambda(lambda x: x[:, BK_WIN-1 :, :])(attention_mul)
        attention_mul = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(attention_mul)
        attention_tensors.append(attention_mul)

    attention_output = keras.layers.concatenate(attention_tensors, axis = -2)

    return attention_output
# attention layer with window size = BKWIN (share paramater)
def attention_layer_share(config, rnn1):
    input_dim = int(rnn1.shape[2])
    batch_size = config['batch_size']
    attention_tensors = []
    # attention_output = rnn1[: 1, :]
    BK_WIN = config['attention_win']
    dense_1 = Dense(BK_WIN, activation='softmax', name='attention')
    permute_1 = Permute((2,1))
    reshape_1 = Reshape((input_dim, BK_WIN))
    for i in range(1, config['maxlen'] + 1, 1):
        if i < BK_WIN - 1:
            rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
            rnn_head = Lambda(lambda x: x[:, 0:1, :])(rnn1)
            rnn_cct = concatenate([rnn_head for i in range(BK_WIN - i)], axis = -2)
            rnn_slice = concatenate([rnn_cct, rnn_slice], axis=-2)
            # attention_mul = merge([a_probs, rnn_slice], mode='mul')
            # attention_mul = Lambda(lambda x: x[:, BK_WIN-1 :, :])(attention_mul)
        elif i == BK_WIN-1:
            rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
            rnn_head = Lambda(lambda x: x[:, 0:1, :])(rnn1)
            rnn_cct = rnn_head
            rnn_slice = concatenate([rnn_cct, rnn_slice], axis=-2)
            # attention_mul = merge([a_probs, rnn_slice], mode='mul')
            # attention_mul = Lambda(lambda x: x[:, BK_WIN-1 :, :])(attention_mul)
        else:
            rnn_slice = Lambda(lambda x : x[:, i - BK_WIN:i, :])(rnn1)

        attr = permute_1(rnn_slice)
        a_1 = reshape_1(attr)
        a_2 = dense_1(a_1)
        a_probs = permute_1(a_2)

        attention_mul = merge([a_probs, rnn_slice], mode='mul')
        # attention_mul = Lambda(lambda x: x[:, BK_WIN-1 :, :])(attention_mul)
        attention_mul = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(attention_mul)
        attention_tensors.append(attention_mul)

    attention_output = keras.layers.concatenate(attention_tensors, axis = -2)

    return attention_output

# attention layer with window size = BKWIN (share paramater)
# http://anthology.aclweb.org/P16-2034
def attention_layer_ACL_simple(config, rnn1):
    input_dim = int(rnn1.shape[2])
    batch_size = config['batch_size']
    attention_tensors = []
    BK_WIN = config['attention_win']
    dense_0 = Dense(1, use_bias=False)
    dense_1 = Dense(1)
    permute_1 = Permute((2,1))
    reshape_1 = Reshape((input_dim, BK_WIN))
    act = Activation('tanh')
    alphas = []
    for i in range(1, config['maxlen'] + 1, 1):
        if i < BK_WIN - 1:
            rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
            rnn_head = Lambda(lambda x: x[:, 0:1, :])(rnn1)
            rnn_cct = concatenate([rnn_head for i in range(BK_WIN - i)], axis = -2)
            rnn_slice = concatenate([rnn_cct, rnn_slice], axis=-2)
        elif i == BK_WIN-1:
            rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
            rnn_head = Lambda(lambda x: x[:, 0:1, :])(rnn1)
            rnn_cct = rnn_head
            rnn_slice = concatenate([rnn_cct, rnn_slice], axis=-2)
        else:
            rnn_slice = Lambda(lambda x : x[:, i - BK_WIN:i, :])(rnn1)
        # if i == 0:
        #     rnn_slice = Lambda(lambda x : x[:, 0:1, :])(rnn1)
        # else:
        #     rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
        # rnn_slice: batch_size * BK_WIN * hidden_size
        a_1 = act(rnn_slice)



        # a_1 = Dropout(0.4)(a_1)
        # a2: batch * timestep * 1
        a_2 = dense_1(a_1)
        alpha = permute_1(a_2)

        # alpha: batch * 1 * timestep
        alpha = Activation("softmax")(alpha)
        tensor = dot([alpha, rnn_slice], [2,1])
        attention_tensors.append(tensor)
        alphas.append(alpha)
    # alpha: 1 * 1 * timestep
    attention_output = keras.layers.concatenate(attention_tensors, axis = -2)
    return attention_output, alphas

# attention layer with window size = BKWIN (share paramater)
# http://anthology.aclweb.org/P16-2034
def attention_layer_ACL_withW(config, rnn1):
    input_dim = int(rnn1.shape[2])
    batch_size = config['batch_size']
    attention_tensors = []
    # attention_output = rnn1[: 1, :]
    BK_WIN = config['attention_win']
    dense_0 = Dense(int(config['hidden_vector']/4))
    dense_1 = Dense(1)
    permute_1 = Permute((2,1))
    reshape_1 = Reshape((input_dim, BK_WIN))
    act = Activation('tanh')
    for i in range(1, config['maxlen'] + 1, 1):
        if i < BK_WIN - 1:
            rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
            rnn_head = Lambda(lambda x: x[:, 0:1, :])(rnn1)
            rnn_cct = concatenate([rnn_head for i in range(BK_WIN - i)], axis = -2)
            rnn_slice = concatenate([rnn_cct, rnn_slice], axis=-2)
        elif i == BK_WIN-1:
            rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
            rnn_head = Lambda(lambda x: x[:, 0:1, :])(rnn1)
            rnn_cct = rnn_head
            rnn_slice = concatenate([rnn_cct, rnn_slice], axis=-2)
        else:
            rnn_slice = Lambda(lambda x : x[:, i - BK_WIN:i, :])(rnn1)
        # rnn_slice: batch_size * BK_WIN * hidden_size
        rnn_slice = dense_0(rnn_slice)
        a_1 = act(rnn_slice)
        # a_1 = Dropout(0.4)(a_1)
        # a2: batch * timestep * 1
        a_2 = dense_1(a_1)
        alpha = permute_1(a_2)
        # alpha: batch * 1 * timestep
        alpha = Activation("softmax")(alpha)
        tensor = dot([alpha, rnn_slice], [2,1])
        attention_tensors.append(tensor)
    # alpha: 1 * 1 * timestep
    attention_output = keras.layers.concatenate(attention_tensors, axis = -2)
    return attention_output


# baseline - general
# https://nlp.stanford.edu/pubs/emnlp15_attn.pdf -- section 3.1
def attention_context_gen_k(config, rnn1):
    hidden_size = int(rnn1.shape[2])
    time_step = config['maxlen']
    BK_WIN = config['attention_win']
    vectors = []
    alphas = []
    for i in range(time_step):
        # slice window for each time step
        
        if i < BK_WIN - 1:
            rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
            rnn_head = Lambda(lambda x: x[:, 0:1, :])(rnn1)
            rnn_cct = concatenate([rnn_head for i in range(BK_WIN - i)], axis = -2)
            rnn_slice = concatenate([rnn_cct, rnn_slice], axis=-2)
            h_t = Lambda(lambda x: x[:, i:i+1, :])(rnn1)
        elif i == BK_WIN-1:
            rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
            rnn_head = Lambda(lambda x: x[:, 0:1, :])(rnn1)
            rnn_cct = rnn_head
            rnn_slice = concatenate([rnn_cct, rnn_slice], axis=-2)
            h_t = Lambda(lambda x: x[:, i:i+1, :])(rnn1)
        else:
            rnn_slice = Lambda(lambda x : x[:, i - BK_WIN:i, :])(rnn1)
            h_t = Lambda(lambda x: x[:, i:i+1, :])(rnn1)
        
        # if i == 0:
        #     rnn_slice = Lambda(lambda x : x[:, 0:1, :])(rnn1)
        #     h_t = Lambda(lambda x: x[:, i:i+1, :])(rnn1)
        # else:
        #     rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
        #     h_t = Lambda(lambda x: x[:, i:i+1, :])(rnn1)

            
        dense_1 = Dense(hidden_size, use_bias=False)
        dense_2 = Dense(hidden_size, use_bias=False, activation='tanh')
        
        # rnn_slice = Reshape((BK_WIN, hidden_size))(rnn_slice)
        h_t = Reshape((1, hidden_size))(h_t)
        # apply attention
        attention_vector, alpha = attention_3d_bocks(config, BK_WIN, rnn_slice, h_t, dense_1, dense_2)
        vectors.append(attention_vector)
        alphas.append(alpha)
    attention_output = keras.layers.concatenate(vectors, axis=-2)
    return attention_output, alphas

# baseline - concatenate
# https://nlp.stanford.edu/pubs/emnlp15_attn.pdf -- section 3.1
def attention_context_concat_k(config, rnn1):
    hidden_size = int(rnn1.shape[2])
    time_step = config['maxlen']
    BK_WIN = config['attention_win']
    vectors = []
    alphas = []
    for i in range(time_step):
        if i < BK_WIN - 1:
            rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
            rnn_head = Lambda(lambda x: x[:, 0:1, :])(rnn1)
            rnn_cct = concatenate([rnn_head for i in range(BK_WIN - i)], axis = -2)
            rnn_slice = concatenate([rnn_cct, rnn_slice], axis=-2)
        elif i == BK_WIN-1:
            rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
            rnn_head = Lambda(lambda x: x[:, 0:1, :])(rnn1)
            rnn_cct = rnn_head
            rnn_slice = concatenate([rnn_cct, rnn_slice], axis=-2)
        else:
            rnn_slice = Lambda(lambda x : x[:, i - BK_WIN:i, :])(rnn1)

        # if i == 0:
        #     rnn_slice = Lambda(lambda x : x[:, 0:1, :])(rnn1)
        #     BK_WIN = 1
        # else:
        #     rnn_slice = Lambda(lambda x : x[:, :i, :])(rnn1)
        #     BK_WIN = i


        h_t = Lambda(lambda x: x[:, i:i+1, :])(rnn1)
        # h_t = Reshape((1, hidden_size))(h_t)
        # rnn_slice = concat_states(config, h_t,rnn_slice, BK_WIN)
        rnn_slice = concat_states(config, h_t,rnn_slice, BK_WIN)

        dense_0 = Dense(hidden_size, use_bias=False, activation='tanh')

        dense_1 = Dense(1, use_bias=False)

        dense_2 = Dense(hidden_size, use_bias=False, activation='tanh')

        attention_vector, alpha = attention_3d_bocks_concat(config, BK_WIN, rnn_slice, h_t, 2*hidden_size, dense_0, dense_1, dense_2)
        vectors.append(attention_vector)
        alphas.append(alpha)
    attention_output = keras.layers.concatenate(vectors, axis=-2)
    return attention_output, alphas



def attention_3d_bocks(config, time_step, hidden_states, h_t, dense_1, dense_2):
    # dense_1 = Dense(BK_WIN, use_bias=False)
    # dense_2 = Dense(hidden_size, use_bias=False, activation='tanh')

    # hidden_size = config['hidden_vector']
    hidden_size = int(hidden_states.shape[2])

    
    # score_first_part: W_alpha * h_s
    # Input shape:  (hidden_size , hidden_size) * (hidden_size, BK_WIN)
    # output shape: hidden_size , BK_WIN

    # dense dimension: hidden_size , hidden_size
    # dense output: BK_WIN, hidden_size
    score_first_part = dense_1(hidden_states)
    score_first_part = Permute((2,1))(score_first_part)
    
    # h_t * (W_alpha * h_s)
    # Input: (1 , hidden_size) * (hidden_size , BK_WIN)
    # output: 1 , BK_WIN
    score = dot([h_t, score_first_part], [2,1])

    # alpha(s): softmax of score
    attention_weights = Activation('softmax')(score)

    
    # attention_weights = Lambda(lambda x: x, name='attention')(attention_weights)

    # c_t = alpha * hidden_states
    # Input: (1 , BK_WIN) * (BK_WIN , hidden_size) 
    # output: 1 , hidden_size
    context_vector = dot([attention_weights, hidden_states], [2,1])

    # [c_t; h_t]
    # Input: (1 , hidden_size) ; (1 , hidden_size)
    # Output: 1 , 2*hidden_size
    pre_activation = concatenate([context_vector, h_t], axis = -1)

    # W_c * [c_t ; h_t]
    # Input: (1 , 2*hidden_size) * (2*hidden_size, hidden_size)
    # Output: 1, hidden_size
    # dense dimension: 2*hidden_size, hidden_size
    attention_vector = dense_2(pre_activation)
    
    # h_t_prime
    # check the shape
    attention_vector = Reshape((1, hidden_size))(attention_vector)
    return attention_vector, attention_weights

def attention_3d_bocks_concat(config, time_step, h_s, h_t, hidden_size, dense_0, dense_1, dense_2):
    # dense_0 = Dense(hidden_size, use_bias=False, activation='tanh')
    # dense_1 = Dense(1, use_bias=False, activation='softmax')
    # dense_2 = Dense(hidden_size, use_bias=False, activation='tanh')

    # tanh(W_a * [h_s, h_t])
    # Input: h_s: BK_WIN, 2*hidden_size  
    # (BK_WIN, 2*hidden_size) * (2*hidden_states, BK_WIN)  
    # output: hidden_states, BK_WIN
    
    # dense_output: BK_WIN, hidden_states
    score_first_part = dense_0(h_s)
    # # output: hidden_states, BK_WIN
    # score_first_part = Permute((2,1))(score_first_part)
    
    # v_a * tanh(...)
    # Input: BK_WIN, hidden_states
    # (1, hidden_states) * (hidden_states, BK_WIN)
    # output: 1, BK_WIN
    # dense dimention(v_a): hidden_states, 1 
    attention_weights = dense_1(score_first_part)
    attention_weights = Permute((2,1))(attention_weights)
    attention_weights = Activation('softmax')(attention_weights)
    attention_weights = Permute((2,1))(attention_weights)

    # (hidden_states,BK_WIN) * (BK_WIN, 1)
    # output: hidden_states, 1
    context_vector = dot([h_s, attention_weights], [1,1])
    # import pdb
    # pdb.set_trace()
    context_vector = Reshape((1,hidden_size))(context_vector) 

    context_vector = concatenate([h_t, context_vector], axis=-1)

    attention_vector = dense_2(context_vector)
    attention_weights = Permute((2,1))(attention_weights)
    return attention_vector, attention_weights



def concat_states(config, h_t, h_s, time_step, hsize=-1):
    if hsize == -1:
        hsize = 2*int(h_s.shape[2])
    res = []
    for i in range(time_step):
        h_s_t = Lambda(lambda x: x[:, i:i+1, :])(h_s)
        con_t = concatenate([h_s_t, h_t], axis=-1)
        res.append(con_t)
    if time_step > 1:
        res = concatenate(res, axis = 1)
    else:
        res = res[-1]
    res = Reshape((time_step, hsize))(res)
    return res

