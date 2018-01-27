import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm

class KmerAutoencoder:
  def __init__(self, num_neurons, num_codes):
    self.num_neurons = num_neurons
    self.num_codes = num_codes
  
  def model(self, X):
    hidden1 = fully_connected(X, self.num_neurons,
                              normalizer_fn=batch_norm,
                              reuse=tf.AUTO_REUSE, scope='hidden1')
    hidden2 = fully_connected(hidden1, self.num_neurons,
                              normalizer_fn=batch_norm,
                              reuse=tf.AUTO_REUSE, scope='hidden2')
    hidden3 = fully_connected(hidden1, self.num_neurons,
                              normalizer_fn=batch_norm,
                              reuse=tf.AUTO_REUSE, scope='hidden3')
    mean = fully_connected(hidden2, self.num_codes,
                           activation_fn=None,
                           reuse=tf.AUTO_REUSE, scope='mean')
    gamma = fully_connected(hidden2, self.num_codes,
                            activation_fn=None,
                            reuse=tf.AUTO_REUSE, scope='gamma')
    sigma = tf.exp(0.5 * gamma)
    noise = tf.random_normal(tf.shape(sigma), dtype=tf.float32)
    codes = mean + sigma * noise
    hidden4 = fully_connected(codes, self.num_neurons,
                              normalizer_fn=batch_norm,
                              reuse=tf.AUTO_REUSE, scope='hidden4')
    hidden5 = fully_connected(codes, self.num_neurons,
                              normalizer_fn=batch_norm,
                              reuse=tf.AUTO_REUSE, scope='hidden5')
    hidden6 = fully_connected(hidden3, self.num_neurons,
                              normalizer_fn=batch_norm,
                              reuse=tf.AUTO_REUSE, scope='hidden6')
    logits = fully_connected(hidden4, 136,
                             activation_fn=None,
                             reuse=tf.AUTO_REUSE, scope='logits')
    outputs = tf.nn.sigmoid(logits)

    latent_loss = 0.5 * tf.reduce_sum(
      tf.exp(gamma) + tf.square(mean) - 1 - gamma)
    reconstruction_loss = tf.reduce_sum(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits))
    return mean,gamma,codes,logits,outputs,latent_loss,reconstruction_loss

  def train(self, X, learning_rate):
    mean,gamma,codes,logits,outputs,ll,rl = self.model(X)
    cost = ll + rl
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return ll,rl,optimizer.minimize(cost)
  
  def infer(self, X):
    _,_,codes,_,_ = self.model(X)
    return codes
