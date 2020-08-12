# seq_max_len = 25
# randomize
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import os
import tensorflow as tf

from edward.models import Bernoulli, Normal, Categorical
from edward.util import Progbar
from tensorflow.contrib import slim

from sentiment_data import *
from review_data import *
import sys

def drop_long_sents(train_exs, seq_max_len):
    list_len = np.array([len(exs.indexed_words) for exs in train_exs])
    list_idx = np.where(list_len <= seq_max_len)[0]
    return [train_exs[idx] for idx in list_idx]

def deconv_output_length(input_length, filter_size, padding, stride):
  """Determines output length of a transposed convolution given input length.
  Arguments:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.
  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  input_length *= stride
  if padding == 'valid':
    input_length += max(filter_size - stride, 0)
  elif padding == 'full':
    input_length -= (stride + filter_size - 2)
  return input_length

def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  """Determines output length of a convolution given input length.
  Arguments:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.
      dilation: dilation rate, integer.
  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  assert padding in {'same', 'valid', 'full'}
  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  if padding == 'same':
    output_length = input_length
  elif padding == 'valid':
    output_length = input_length - dilated_filter_size + 1
  elif padding == 'full':
    output_length = input_length + dilated_filter_size - 1
  return (output_length + stride - 1) // stride

def pad_to_length(np_arr, length, constant_value=0):
    result = constant_value * np.ones(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

def pre_pad_to_length(np_arr, length, constant_value=0):
    '''
    we actually should pad the index with "UNK" i.e. zero embedding
    '''
    result = constant_value * np.ones(length)
    result[-np_arr.shape[0]:] = np_arr
    return result

def generator(array, batch_size):
    """Generate batch with respect to array's first axis.
    """
    start = 0  # pointer to where we are in iteration
    while True:
        stop = start + batch_size
        diff = stop - array.shape[0]
        if diff <= 0:
            batch = array[start:stop]
            start += batch_size
        else:
            batch = np.concatenate((array[start:], array[:diff]))
            start = diff
#         batch = batch.astype(np.float32) / 255.0  # normalize pixel intensities
#         batch = np.random.binomial(1, batch)  # binarize images
#         #     print(batch.shape)
        yield batch
def generative_network_train(z, x_input_id, M, hidden_vec_size, latent_vec_size, feat_vec_size, seq_max_len, word_embeddings):# feat_vec_size = hidden_vec_size = 50
    """Generative network to parameterize generative model. It takes
    latent variables as input and outputs the likelihood parameters.
    logits = neural_network(z)
    """
    with tf.variable_scope("gen"):
        #z: [M, latent_vec_size]
        x_vectors = tf.nn.embedding_lookup(word_embeddings, x_input_id) # [M, seq_max_len, feat_vec_size]
        # add noise--------------------
        noise_vectors = tf.convert_to_tensor(np.random.randn(M, seq_max_len, feat_vec_size)*1e-2, dtype=tf.float32)
        x_vectors = x_vectors + noise_vectors
        #-----------------------------
#         x_vectors = tf.slice(x_vectors, [0,1,0], [-1,-1,-1])
    #     x_bos = tf.concat([])
    # UNK: CHANGE TO WORD_EMBEEDINGS[VOC_SIZE]
        z = tf.reshape(z, [M, 1, latent_vec_size])
        z = tf.tile(z, [1, seq_max_len, 1]) # [M,25,latent_vec_size] # repeat z in axis 1
        # concat z and the input word vectors
        zx = tf.concat([z, x_vectors], axis = 2) # [M, 25, latent_vec_size + feat_vec_size]

        h1_vec_size = hidden_vec_size #50
        h2_vec_size = hidden_vec_size
        h3_vec_size = hidden_vec_size
        h4_vec_size = hidden_vec_size #50
        h5_vec_size = hidden_vec_size
        h6_vec_size = hidden_vec_size

        zx = tf.pad(zx, [[0,0],[4,0],[0,0]], "CONSTANT") # [M, 29, h1_vec_size]

        filters_1 = tf.get_variable('gconv_f1', [5, feat_vec_size + latent_vec_size, h1_vec_size], 
                                   initializer=tf.contrib.layers.xavier_initializer(seed=0))
        h1 = tf.nn.relu(tf.nn.conv1d(zx, filters_1, stride = 1, padding = 'VALID')) #[M, 25, h1_vec_size]
        h1 = tf.pad(h1, [[0,0],[4,0],[0,0]], "CONSTANT") # [M, 29, h1_vec_size]

        filters_2 = tf.get_variable('gconv_f2', [5, h1_vec_size, h2_vec_size], 
                                   initializer=tf.contrib.layers.xavier_initializer(seed=0))
        h2 = tf.nn.relu(tf.nn.conv1d(h1, filters_2, stride = 1, padding = 'VALID')) #[M, 25, h1_vec_size]
        h2 = tf.pad(h2, [[0,0],[4,0],[0,0]], "CONSTANT")  # [M, 29, h2_vec_size]

        filters_3 = tf.get_variable('gconv_f3', [5, h2_vec_size, h3_vec_size], 
                                   initializer=tf.contrib.layers.xavier_initializer(seed=0))
        h3 = tf.nn.relu(tf.nn.conv1d(h2, filters_3, stride = 1, padding = 'VALID')) #[M, 25, h1_vec_size]
        h3 = tf.pad(h3, [[0,0],[4,0],[0,0]], "CONSTANT") # [M, 29, h3_vec_size]

        filters_4 = tf.get_variable('gconv_f4', [5, h3_vec_size, h4_vec_size], 
                                   initializer=tf.contrib.layers.xavier_initializer(seed=0))
        h4 = tf.nn.relu(tf.nn.conv1d(h3, filters_4, stride = 1, padding = 'VALID')) #[M, 25, h1_vec_size]
        h4 = tf.pad(h4, [[0,0],[4,0],[0,0]], "CONSTANT") # [M, 29, h4_vec_size]

        filters_5 = tf.get_variable('gconv_f5', [5, h4_vec_size, h5_vec_size], 
                                   initializer=tf.contrib.layers.xavier_initializer(seed=0))
        h5 = tf.nn.relu(tf.nn.conv1d(h4, filters_5, stride = 1, padding = 'VALID')) #[M, 25, h1_vec_size]
        h5 = tf.pad(h5, [[0,0],[4,0],[0,0]], "CONSTANT") # [M, 29, h5_vec_size]

        filters_6 = tf.get_variable('gconv_f6', [5, h5_vec_size, h6_vec_size], 
                                   initializer=tf.contrib.layers.xavier_initializer(seed=0))
        h6 = tf.nn.relu(tf.nn.conv1d(h5, filters_6, stride = 1, padding = 'VALID')) #[M, 25, h1_vec_size]


        logits = tf.tensordot(h6, tf.transpose(word_embeddings), 1) # [M, 25, vocabulary_size]

    return logits # [M, 25, vocabulary_size]

# def generative_network_test(z, x_input_id, M, hidden_vec_size, latent_vec_size, seq_max_len, word_embeddings):
def inference_network(x, M, hidden_vec_size, latent_vec_size, feat_vec_size): # [M, seq_max_len, feat_vec_size]
    """Inference network to parameterize variational model. It takes
    data as input and outputs the variational parameters.
    loc, scale = neural_network(x)
    """
    # x: [M, seq_max_len, feat_vec_size]
            
    h1_vec_size = hidden_vec_size #50
    h2_vec_size = hidden_vec_size
    h3_vec_size = hidden_vec_size
    h4_vec_size = hidden_vec_size #50
    h5_vec_size = hidden_vec_size
    h6_vec_size = hidden_vec_size
        
    filters_1 = tf.get_variable('conv_f1', [5, feat_vec_size, h1_vec_size], 
                               initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h1 = tf.nn.relu(tf.nn.conv1d(x, filters_1, stride = 1, padding = 'VALID')) #[M, 21, h1_vec_size]
    
    filters_2 = tf.get_variable('conv_f2', [5, h1_vec_size, h2_vec_size], 
                               initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h2 = tf.nn.relu(tf.nn.conv1d(h1, filters_2, stride = 1, padding = 'VALID')) #[M, 17, h1_vec_size]
    
    filters_3 = tf.get_variable('conv_f3', [5, h2_vec_size, h3_vec_size], 
                               initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h3 = tf.nn.relu(tf.nn.conv1d(h2, filters_3, stride = 1, padding = 'VALID')) #[M, 13, h1_vec_size]

    filters_4 = tf.get_variable('conv_f4', [5, h3_vec_size, h4_vec_size], 
                               initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h4 = tf.nn.relu(tf.nn.conv1d(h3, filters_4, stride = 1, padding = 'VALID')) #[M, 9, h1_vec_size]
    
    filters_5 = tf.get_variable('conv_f5', [5, h4_vec_size, h5_vec_size], 
                               initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h5 = tf.nn.relu(tf.nn.conv1d(h4, filters_5, stride = 1, padding = 'VALID')) #[M, 5, h1_vec_size]
    
    filters_6 = tf.get_variable('conv_f6', [5, h5_vec_size, h6_vec_size], 
                               initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h6 = tf.nn.relu(tf.nn.conv1d(h5, filters_6, stride = 1, padding = 'VALID')) #[M, 1, h1_vec_size]
    
    h6 = tf.reshape(h6, [M, -1])
    h6 = tf.concat([h6, tf.ones([M, 1])], axis = 1)
    # final_hidden_vec_size = h3.shape[1]
    weights_inf_2_1 = tf.get_variable("w_inf_2_1", [h6_vec_size + 1, latent_vec_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
    weights_inf_2_2 = tf.get_variable("w_inf_2_2", [h6_vec_size + 1, latent_vec_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
    loc = tf.tensordot(h6, weights_inf_2_1, 1)
    scale = tf.nn.softplus(tf.tensordot(h6, weights_inf_2_2, 1))
    return loc, scale # [M, latent_vec_size]


def train_conv(train_exs, word_vectors, n_epoch = 1, batch_size = 1):
    # ------------define constants and train_mat
    

    vocab_size = word_vectors.vectors.shape[0]  
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 25
    train_exs = drop_long_sents(train_exs, seq_max_len)
    num_train_samples = len(train_exs)
    
    feat_vec_size = word_vectors.get_embedding("UNK").shape[0]
    
    M = batch_size # batch size during training
    d = 32  # latent dimension
    latent_vec_size = d

#     hidden_vec_size = min(feat_vec_size, 100) 
    hidden_vec_size = 50
    
    print('vocabulary size = %d'%vocab_size)
    print('seq_max_len = %d'%seq_max_len)
    print('num_train_samples = %d'%num_train_samples)
    print('feat_vec_size = %d'%feat_vec_size)
    
    print('batch_size = %d'%M)
    print('hidden_vec_size = %d'%hidden_vec_size)
    print('latent_vec_size = %d'%latent_vec_size)
    
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len, vocab_size-1) 
                                for ex in train_exs], dtype=np.int32)
    
    # MODEL(Decoder) (from z to x)
    word_embeddings = tf.convert_to_tensor(word_vectors.vectors[:-1,:], dtype=tf.float32) 
    # delete 'unk'(not good for zero vector)

    unk_bos_var = tf.get_variable("unk_bos_emb", shape = (2, feat_vec_size), dtype = tf.float32,                                 
                                  initializer=tf.contrib.layers.xavier_initializer())

    word_embeddings_new = tf.concat([word_embeddings, unk_bos_var], axis = 0) 
    # concact 'unk' and 'bos' (variables, learned from training process)


    z = Normal(tf.zeros([M, d]), tf.ones([M, d]))

    x_id  = tf.placeholder(tf.int32, shape=[M, seq_max_len]) # including '.', no bos tags.

    x_input_id = tf.slice(x_id, [0,0],[-1,seq_max_len - 1]) # delete '.'
    # concat index of 'BOS' in the word_embeddings_new at the beginning of each sentence
    # assume word_embeddings have W(vocab_size) words after deleting UNK. 
    # Adding unk, bos gives W+2, 
    # bos is the W+1-th, which is just equal to orginial word_vectors.vectors.shape[0]
    x_input_id = tf.concat([ tf.ones([M,1], dtype=tf.int32)*word_vectors.vectors.shape[0], x_input_id], axis = 1)

    logits = generative_network_train(z, x_input_id, M, hidden_vec_size, latent_vec_size, 
                                      feat_vec_size, seq_max_len, word_embeddings_new)

    x = Categorical(logits = logits) # shape: [M, 60, vocabulary_size]    

# INFERENCE(Encoder) (from x to z)
    #x_id = tf.placeholder(tf.int32, shape=[M, seq_max_len]) # [M,25] #input_word_indices

    x_input = tf.nn.embedding_lookup(word_embeddings_new, x_id) # [M, 25, 50]

    loc, scale = inference_network(x_input, M, hidden_vec_size, latent_vec_size, feat_vec_size)

    qz = Normal(loc, scale)

    #------------------------------
    # BIND p(x, z) and q(z | x) to the same placeholder for x.
    data = {x: x_id}
    inference = ed.KLqp({z: qz}, data)
    
    
    # Decay the learning rate by a factor of 0.99 every 1000 gradient steps (for larger datasets you'll want a slower
    # weight decay schedule
    
    decay_steps = 100
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.001
    

    print('initial_learning_rate = %.3f'%initial_learning_rate)
    print('learning_rate_decay_factor = %.2f'%learning_rate_decay_factor)
    print('decay_steps = %d'%decay_steps)
    
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    
    optimizer = tf.train.AdamOptimizer(lr, epsilon=1.0)    
    
    
    inference.initialize(optimizer=optimizer, logdir='log')
        # Logging with Tensorboard
    
    
    


    n_iter_per_epoch = train_mat.shape[0] // M

    x_train_generator = generator(train_mat, M)
    # x_train_len_generator = generator(train_seq_lens, M)
    
    init_op = tf.global_variables_initializer()
    #tf.global_variables_initializer().run()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()


    with tf.Session() as sess:
        
        sess.run(init_op)

        for epoch in range(1, n_epoch + 1):
            print("Epoch: {0}".format(epoch))
            avg_loss = 0.0

            pbar = Progbar(n_iter_per_epoch)
            for t in range(1, n_iter_per_epoch + 1):
                pbar.update(t)

                x_batch = next(x_train_generator)
                #         x_len_batch = next(x_train_len_generator)

                info_dict = inference.update(feed_dict={x_id: x_batch})

                avg_loss += info_dict['loss']

                # Print a lower bound to the average marginal likelihood for an
                # image.
            avg_loss = avg_loss / n_iter_per_epoch
            avg_loss = avg_loss / M
            print("-log p(x) <= {:0.3f}".format(avg_loss))

        save_path = saver.save(sess, "tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)
        
    return info_dict, avg_loss, logits, x, z, qz


if __name__ == '__main__':    
 
    #----------load data-----------
    # Use either 50-dim or 300-dim vectors
    word_vectors = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    #word_vectors = read_word_embeddings("data/glove.6B.300d-relativized.txt")

    # Load train exs and tokenize
    #train_exs = read_and_index_sentiment_examples("data/train.txt", word_vectors.word_indexer)
    train_exs = read_and_index_review_examples("data/review.json", word_vectors.word_indexer, 200000)
    # print(repr(len(train_exs)) + " train examples")
    
    if len(sys.argv) >= 3:
        
        n_epoch = int(sys.argv[1])
        batch_size = int(sys.argv[2])
        
        info_dict, avg_loss, logits, x, z, qz = train_conv(train_exs, word_vectors, n_epoch, batch_size)
        
        # write outputs 
#        out_dir = "./tmp/out"
#         if not os.path.exists(out_dir):
#             os.makedirs(out_dir)
#         f = open(out_dir + '/output_sents_x.txt','w')
#         M = batch_size
#         for i in range(5):
#             for m in range(M):
#                 sent = [word_vectors.word_indexer.get_object(idx) for idx in x.eval()[m]]
#                 mysent = ' '.join(sent) + ".\n"
#                 f.write(mysent)
#         f.close()
 
#         f = open(out_dir + '/output_sents_y.txt','w')
#         y = tf.argmax(logits, axis = 2)
        
#         for i in range(5):
#             for m in range(M):
#                 sent = [word_vectors.word_indexer.get_object(idx) for idx in y.eval()[m]]
#                 mysent = ' '.join(sent) + " .\n"
#                 f.write(mysent)
#         f.close()

    else:
        raise Exception("Pass in n_epoch run the appropriate system !")
