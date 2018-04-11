import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from math import pi

class KmerAutoencoder:
  def __init__(self, num_neurons, num_hidden_layers, num_codes):
    self.num_neurons = num_neurons
    self.num_hidden_layers = num_hidden_layers
    self.num_codes = num_codes
  
  def model(self, X):
    def res_full(input, n_out, scope):
      out = fully_connected(input, n_out,
                            activation_fn=tf.tanh,
                            normalizer_fn=batch_norm,
                            reuse=tf.AUTO_REUSE,
                            scope=scope)
      return tf.add(input, out)
    
    kmer_sizes = [2, 10, 32, 136, 512, 2080] 
    layer_sizes = [10, 20, 50, 200, 1000, 3000]
    scopes = ['monomer', 'dimer', 'trimer',
              'tetramer', 'pentamer', 'hexamer']
    mers = [X[:, :2], X[:, 2:12], X[:, 12:44],
            X[:, 44:180], X[:, 180:692] ,X[:, 692:2772]]
    kmer_encoder_outputs = []
    for size, scope, mer in zip(layer_sizes, scopes, mers):
      kmer_encoder_output = fully_connected(mer, size,
                                  activation_fn=None,
                                  reuse=tf.AUTO_REUSE,
                                  scope=scope+'_encoder_0')
      for n_layer in range(1,4):
        kmer_encoder_output = res_full(kmer_encoder_output, size,
                                    scope=scope+'_encoder_'+str(n_layer))
      kmer_encoder_output = fully_connected(kmer_encoder_output,
                        self.num_neurons,
                        activation_fn=None,
                        reuse=tf.AUTO_REUSE,
                        scope=scope+'_encoder_'+str(n_layer+1)))
      kmer_layer_outputs.append(kmer_encoder_output)
    
    encoder_out = tf.add_n(kmer_layer_outputs)
    for n_layer in range(3):
      encoder_out = res_full(encoder_out, self.num_neurons,
                             scope='encoder_'+str(n_layer))
    
    mean = fully_connected(encoder_out, self.num_codes,
                           activation_fn=None,
                           reuse=tf.AUTO_REUSE, scope='mean')
    gamma = fully_connected(encoder_out, self.num_codes,
                            activation_fn=None,
                            reuse=tf.AUTO_REUSE, scope='gamma')
    sigma = tf.exp(0.5 * gamma)
    noise = tf.random_normal(tf.shape(sigma), dtype=tf.float32)
    codes = mean + sigma * noise
    
    decoder_out = fully_connected(codes, self.num_neurons,
                                  activation_fn=None,
                                  reuse=tf.AUTO_REUSE,
                                  scope='decoder_0')
    for n_layer in range(1,4):
      decoder_out = res_full(decoder_out, self.num_neurons,
                             scope='decoder_'+str(n_layer))
    kmer_decoder_outputs = []
    for kmer_size, layer_size, scope in zip(kmer_sizes, layers_sizes, scopes):
      kmer_decoder_output = fully_connected(decoder_out, size,
                                            activation_fn=None,
                                            reuse=tf.AUTO_REUSE,
                                            scope=scope+'_decoder_0')
      for n_layer in range(1,4):
        kmer_decoder_output = res_full(kmer_decoder_output, size,
                                       scope=scope+'_decoder_'+str(n_layer))
        kmer_decoder_output = fully_connected(kmer_decoder_output, kmer_size,
                                       activation_fn=None,
                                       reuse=tf.AUTO_REUSE,
                                       scope=scope+'_decoder_'+str(n_layer+1))
      kmer_decoder_outputs.append(kmer_decoder_output)
    decoder_out = tf.concat(kmer_decoder_outputs)
    
    logits = fully_connected(decoder_out, sum(kmer_sizes),
                             activation_fn=None,
                             reuse=tf.AUTO_REUSE, scope='logits')
    o_mean = tf.nn.sigmoid(logits)
    o_gamma = fully_connected(decoder_out, sum(kmer_sizes),
                              activation_fn=None,
                              reuse=tf.AUTO_REUSE, scope='o_gamma')
    o_sigma = tf.exp(0.5 * o_gamma)
    normal_dist = tf.distributions.Normal(o_mean, o_sigma) 
    outputs = normal_dist.mode()
    reconstruction_loss = -tf.reduce_sum(normal_dist.log_prob(X))

    latent_loss = 0.5 * tf.reduce_sum(
      tf.exp(gamma) + tf.square(mean) - 1 - gamma)

    return mean,gamma,codes,None,outputs,latent_loss,reconstruction_loss

  def train(self, X, learning_rate):
    mean,gamma,codes,logits,outputs,ll,rl = self.model(X)
    cost = ll + rl
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return ll,rl,optimizer.minimize(cost)
  
  def infer(self, X):
    _,_,codes,_,_ = self.model(X)
    return codes
