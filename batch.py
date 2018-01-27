import tensorflow as tf
from kmer import KmerCounter

class DnaSequenceBatch():
  def __init__(self, generator, batch_size):
    self.dataset = tf.data.Dataset.from_generator(generator,
                                                  tf.float32,
                                                  [None])
    self.batch = self.dataset.batch(batch_size)
    self.batch_iterator = self.batch.make_initializable_iterator()
  
  def get_next_batch(self):
    return self.batch_iterator.get_next()

if __name__ == '__main__':
  l = ['ATGCGTGCTA',
       'AATAAGTGCC',
       'AATTCCCGGG',
       'TTGGATGCGC'] 

  kmer_counter = KmerCounter(k=4)
  def gen():
    for s in l:
      yield kmer_counter.count(s)

  with tf.Session() as sess:
    dataset = DnaSequenceBatch(gen, 2)
    sess.run(dataset.batch_iterator.initializer)
    print(sess.run(dataset.get_next_batch()))
    print(sess.run(dataset.get_next_batch()))
