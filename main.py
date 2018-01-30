#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import time
from model import KmerAutoencoder

'''
def data_generator(f):
  def gen():
    while True:
      f.seek(0)
      for s in f:
        r1 = np.array([float(n) for n in s.strip().split()[1:]],
                      dtype=np.float32)
        r2 = np.array([float(n) for n in next(f).strip().split()[1:]],
                      dtype=np.float32)
        yield (r1+r2)/np.sum(r1+r2)
  return gen
'''
def data_generator(f):
  l = []
  for s in f:
    l.append(np.array([float(n) for n in s.strip().split()[2:]],
                      dtype=np.float32))
  def gen():
    while True:
      for mer in l:
        yield mer
  return gen


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('train_kmer_file', type=argparse.FileType('r'))
parser.add_argument('test_kmer_file', type=argparse.FileType('r'))
parser.add_argument('test_codes_out', type=str)
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()

dataset = tf.data.Dataset.from_generator(data_generator(args.train_kmer_file),
                                         tf.float32,
                                         [136])
batch = dataset.batch(batch_size=args.batch_size)
batch_iterator = batch.make_one_shot_iterator()

test_kmer = []
test_read_name = []
test_taxon = []
args.test_kmer_file.seek(0)
for s in args.test_kmer_file:
  l = s.strip().split()
  test_read_name.append(l[0])
  test_taxon.append(l[1])
  r1 = np.array([ float(n) for n in l[2:]], dtype=np.float32)
  test_kmer.append(r1)
  '''
  r2 = np.array([ float(n) 
                  for n in next(args.test_kmer_file).strip().split()[2:]],
                dtype=np.float32)
  test_kmer.append((r1+r2)/np.sum(r1+r2))
  '''
test_kmer = np.array(test_kmer)
test_size = len(test_read_name)

KA = KmerAutoencoder(num_neurons=500, num_hidden_layers=8, num_codes=10)
ll,rl,opt = KA.train(batch_iterator.get_next(), learning_rate=0.001)

_,_,test_codes,_,_,test_ll,test_rl = KA.model(test_kmer)

init = tf.global_variables_initializer()
num_batch = 0
with tf.Session() as sess:
  init.run()
  begin = time.clock()
  while num_batch < 10000:
    try:
      latent_loss, reconst_loss, _ = sess.run([ll,rl,opt])
    except tf.errors.OutOfRangeError:
      break
    num_batch += 1
    if num_batch % 100 == 0:
      print('train_loss', latent_loss/args.batch_size,
                    reconst_loss/args.batch_size,
                    (latent_loss+reconst_loss)/args.batch_size, flush=True)
      print('batch', num_batch, time.clock() - begin, flush=True)
      begin = time.clock()
      
      tes_c,tes_ll,tes_rl = sess.run([test_codes, test_ll, test_rl])
      codes = [ list(c) for c in tes_c ]
      print('test_loss', tes_ll/test_size, tes_rl/test_size)
      code_file = open(args.test_codes_out+'_'+str(num_batch), 'w')
      for name, taxon, c in zip(test_read_name, test_taxon, codes):
        print('\t'.join([name, taxon] + list(map(str,c))),file=code_file)
      code_file.close()
