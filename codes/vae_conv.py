from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import os
import tensorflow as tf

from edward.models import Bernoulli, Normal, Categorical
from edward.util import Progbar
from observations import mnist
from scipy.misc import imsave
from tensorflow.contrib import slim
import sys

from sentiment_data import *

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
       Without randomize???
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


def generative_network(z, M, hidden_vec_size, latent_vec_size, word_embeddings):# feat_vec_size = hidden_vec_size = 50
    """Generative network to parameterize generative model. It takes
    latent variables as input and outputs the likelihood parameters.
    logits = neural_network(z)
    """

# print(conv_output_length(60, 6, 'valid', 3)) #19
# print(conv_output_length(19, 3, 'valid', 2)) # 9
# print(conv_output_length(9, 3, 'valid', 2)) # 4

# print(deconv_output_length(4,3, 'valid', 2)) #9
# print(deconv_output_length(9,3,'valid',2)) # 19
# print(deconv_output_length(19,6,'valid',3)) # 60

    h1_vec_size = hidden_vec_size #50
    h2_vec_size = hidden_vec_size
    h3_vec_size = hidden_vec_size
    f1_width = 3; stride_1 = 2
    f2_width = 3; stride_2 = 2
    f3_width = 6; stride_3 = 3

    z = tf.reshape(z, [M, 1, latent_vec_size])
    weights_gen_1 = tf.get_variable("w_gen_1", [latent_vec_size,4, latent_vec_size],
                              initializer=tf.contrib.layers.xavier_initializer())
    z = tf.tensordot(z, weights_gen_1, 1)
    
#     z = tf.reshape(z, [M, 1, 1, latent_vec_size])
    output_shape_1 = tf.constant([M, 1, 9, h1_vec_size]) # h1_vec_size = 50
    stride_1 = [1,1,2,1]
    filters_1 = tf.get_variable('deconv_f1', [1, f1_width, h1_vec_size, latent_vec_size],
                               initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h1 = tf.nn.relu(tf.nn.conv2d_transpose(z, filters_1, output_shape_1, stride_1, padding = 'VALID'))
    
    output_shape_2 = tf.constant([M, 1, 19, h2_vec_size])
    stride_2 = [1,1,2,1]
    filters_2 = tf.get_variable('deconv_f2', [1, f2_width, h2_vec_size, h1_vec_size],
                               initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h2 = tf.nn.relu(tf.nn.conv2d_transpose(h1, filters_2, output_shape_2, stride_2, padding = 'VALID'))
    
    output_shape_3 = tf.constant([M, 1, 60, h3_vec_size])
    stride_3 = [1,1,3,1]
    filters_3 = tf.get_variable('deconv_f3', [1, f3_width, h3_vec_size, h2_vec_size],
                               initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h3 = tf.nn.relu(tf.nn.conv2d_transpose(h2, filters_3, output_shape_3, stride_3, padding = 'VALID'))
    
    h3 = tf.reshape(h3, [M, 60, h3_vec_size])
    
    logits = tf.tensordot(h3, tf.transpose(word_embeddings), 1) # [M, 60, vocabulary_size]
   
    return logits # [M, 60, vocabulary_size]

# INFERENCE(Encoder) (from x to z)
def inference_network(x, M, hidden_vec_size, latent_vec_size, feat_vec_size): # [M, seq_max_len, feat_vec_size]
    """Inference network to parameterize variational model. It takes
    data as input and outputs the variational parameters.
    loc, scale = neural_network(x)
    """
    h1_vec_size = hidden_vec_size # 50
    h2_vec_size = hidden_vec_size
    h3_vec_size = hidden_vec_size
    final_hidden_vec_size = 200
#     latent_vec_size = 32
    f1_width = 6; stride_1 = 3
    f2_width = 3; stride_2 = 2
    f3_width = 3; stride_3 = 2
    
    filters_1 = tf.get_variable('conv_f1', [f1_width, feat_vec_size, h1_vec_size], 
                               initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h1 = tf.nn.relu(tf.nn.conv1d(x, filters_1, stride = stride_1, padding = 'VALID'))
    
    filters_2 = tf.get_variable('conv_f2', [f2_width, h1_vec_size, h2_vec_size], 
                               initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h2 = tf.nn.relu(tf.nn.conv1d(h1, filters_2, stride = stride_2, padding = 'VALID'))
    
    filters_3 = tf.get_variable('conv_f3', [f3_width, h2_vec_size, h3_vec_size], 
                               initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h3 = tf.nn.relu(tf.nn.conv1d(h2, filters_3, stride = stride_3, padding = 'VALID'))
    h3 = tf.reshape(h3, [M, -1])
    
    h4 = tf.pad(h3, [[0,0],[0,1]], constant_values=1.0)
    # final_hidden_vec_size = h3.shape[1]
    weights_inf_2_1 = tf.get_variable("w_inf_2_1", [final_hidden_vec_size + 1, latent_vec_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
    weights_inf_2_2 = tf.get_variable("w_inf_2_2", [final_hidden_vec_size + 1, latent_vec_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
    loc = tf.tensordot(h4, weights_inf_2_1, 1)
    scale = tf.nn.softplus(tf.tensordot(h4, weights_inf_2_2, 1))
    return loc, scale


def train_conv(train_exs, n_epoch = 1, batch_size = 2):
    # ------------define constants and train_mat
    num_train_samples = len(train_exs)

    vocab_size = word_vectors.vectors.shape[0]  
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    feat_vec_size = word_vectors.get_embedding("UNK").shape[0]
    
    M = batch_size  # batch size during training
    d = 32  # latent dimension
    latent_vec_size = d

#     hidden_vec_size = min(feat_vec_size, 100) 
    hidden_vec_size = 50
    
    print('vocabulary size = %d'%vocab_size)
    print('seq_max_len = %d'%seq_max_len)
    print('feat_vec_size = %d'%feat_vec_size)
    
    print('batch_size = %d'%M)
    print('hidden_vec_size = %d'%hidden_vec_size)
    print('latent_vec_size = %d'%latent_vec_size)
    
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len, vocab_size-1) 
                                for ex in train_exs], dtype=np.int32)
    
    # MODEL(Decoder) (from z to x)
    word_embeddings = tf.convert_to_tensor(word_vectors.vectors, dtype=tf.float32)

    z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d]))

    logits = generative_network(z, M, hidden_vec_size, latent_vec_size, word_embeddings) # [M, 60, vocabulary_size]

    x = Categorical(logits = logits) # shape: [M, 60, vocabulary_size] 

    # INFERENCE(Encoder) (from x to z)
    x_id = tf.placeholder(tf.int32, shape=[M, seq_max_len]) # [M,60] #input_word_indices

    x_input = tf.nn.embedding_lookup(word_embeddings, x_id)

    loc, scale = inference_network(x_input, M, hidden_vec_size, latent_vec_size, feat_vec_size)

    qz = Normal(loc=loc, scale=scale)

    #------------------------------
    # BIND p(x, z) and q(z | x) to the same placeholder for x.
    data = {x: x_id}
    inference = ed.KLqp({z: qz}, data)
    
    optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
    
    
    inference.initialize(optimizer=optimizer)
    tf.global_variables_initializer().run()

    n_iter_per_epoch = train_mat.shape[0] // M

    x_train_generator = generator(train_mat, M)
    # x_train_len_generator = generator(train_seq_lens, M)

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
   
    return info_dict, logits, x, avg_loss


if __name__ == '__main__':    
 
    #----------load data-----------
    # Use either 50-dim or 300-dim vectors
    word_vectors = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    #word_vectors = read_word_embeddings("data/glove.6B.300d-relativized.txt")

    # Load train exs and tokenize
    train_exs = read_and_index_sentiment_examples("data/train.txt", word_vectors.word_indexer)
    print(repr(len(train_exs)) + " train examples")
    
    if len(sys.argv) >= 3:
        
        n_epoch = int(sys.argv[1])
        batch_size = int(sys.argv[2])
        
        info_dict, logits, x, avg_loss = train_conv(train_exs, n_epoch, batch_size)
        
        # write outputs 
        out_dir = "./tmp/out"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        f = open(out_dir + '/output_sents_x.txt','w')
        M = batch_size
        for i in range(5):
            for m in range(M):
                sent = [word_vectors.word_indexer.get_object(idx) for idx in x.eval()[m]]
                mysent = ' '.join(sent) + ".\n"
                f.write(mysent)
        f.close()

        f = open(out_dir + '/output_sents_y.txt','w')
        y = tf.argmax( tf.nn.softmax(logits), axis = 2)
        
        for i in range(5):
            for m in range(M):
                sent = [word_vectors.word_indexer.get_object(idx) for idx in y.eval()[m]]
                mysent = ' '.join(sent) + " .\n"
                f.write(mysent)
        f.close()
    else:
        raise Exception("Pass in n_epoch run the appropriate system !")
