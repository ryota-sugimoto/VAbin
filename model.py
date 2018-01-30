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
                            reuse=tf.AUTO_REUSE, scope=scope)
      return tf.add(input, out)
    
    encoder_out = fully_connected(X, self.num_neurons,
                                  activation_fn=tf.tanh,
                                  normalizer_fn=batch_norm,
                                  reuse=tf.AUTO_REUSE,
                                  scope='encoder1')
    for n_layer in range(2, self.num_hidden_layers+1):
      encoder_out = res_full(encoder_out, 
                             self.num_neurons, scope='encoder'+str(n_layer))
    
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
                                  activation_fn=tf.tanh,
                                  normalizer_fn=batch_norm,
                                  reuse=tf.AUTO_REUSE,
                                  scope='decoder1')
    for n_layer in range(2, self.num_hidden_layers+1):
      decoder_out = res_full(decoder_out,
                             self.num_neurons, scope='decoder'+str(n_layer))

    logits = fully_connected(decoder_out, 136,
                             activation_fn=None,
                             reuse=tf.AUTO_REUSE, scope='logits')
    o_mean = tf.nn.sigmoid(logits)
    o_gamma = fully_connected(decoder_out, 136,
                              activation_fn=None,
                              reuse=tf.AUTO_REUSE, scope='o_gamma')
    o_sigma = tf.exp(0.5 * o_gamma)
    

    latent_loss = 0.5 * tf.reduce_sum(
      tf.exp(gamma) + tf.square(mean) - 1 - gamma)

    reconstruction_loss = tf.reduce_sum(
      0.5*tf.log(2*pi) + 0.5*o_gamma \
      + tf.losses.mean_squared_error(labels=X, predictions=o_mean)/(2*o_sigma**2))
    return mean,gamma,codes,logits,o_mean,latent_loss,reconstruction_loss

  def train(self, X, learning_rate):
    mean,gamma,codes,logits,outputs,ll,rl = self.model(X)
    cost = ll + rl
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return ll,rl,optimizer.minimize(cost)
  
  def infer(self, X):
    _,_,codes,_,_ = self.model(X)
    return codes
