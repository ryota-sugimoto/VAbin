#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import time
from model import KmerAutoencoder

def data_generator(f):
  f.seek(0)
  def gen():
    for s in f:
      r1 = np.array([float(n) for n in s.strip().split()[1:]],
                    dtype=np.float32)
      r2 = np.array([float(n) for n in next(f).strip().split()[1:]],
                    dtype=np.float32)
      yield (r1+r2)/np.sum(r1+r2)
  return gen

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('train_kmer_file', type=argparse.FileType('r'))
parser.add_argument('test_kmer_file', type=argparse.FileType('r'))
parser.add_argument('test_codes', type=argparse.FileType('w'))
parser.add_argument('--batch_size', type=int, default=1000)
args = parser.parse_args()

dataset = tf.data.Dataset.from_generator(data_generator(args.train_kmer_file),
                                         tf.float32,
                                         [136])
batch = dataset.batch(batch_size=args.batch_size)
batch_iterator = batch.make_one_shot_iterator()

test_kmer = []
test_read_name = []
test_taxon = []
for s in args.test_kmer_file:
  l = s.strip().split()
  test_read_name.append(l[0])
  test_taxon.append(l[1])
  r1 = np.array([ float(n) for n in l[3:]], dtype=np.float32)
  r2 = np.array([ float(n) 
                  for n in next(args.test_kmer_file).strip().split()[3:]],
                dtype=np.float32)
  test_kmer.append((r1+r2)/np.sum(r1+r2))
test_kmer = np.array(test_kmer)
test_size = len(test_read_name)

KA = KmerAutoencoder(num_neurons=500, num_codes=40)
ll,rl,opt = KA.train(batch_iterator.get_next(), learning_rate=0.001)

_,_,test_codes,_,_,test_ll,test_rl = KA.model(test_kmer)

init = tf.global_variables_initializer()
num_batch = 0
with tf.Session() as sess:
  init.run()
  begin = time.clock()
  while True:
    try:
      latent_loss, reconst_loss, _ = sess.run([ll,rl,opt])
    except tf.errors.OutOfRangeError:
      break
    num_batch += 1
    if num_batch % 100 == 0:
      print('loss', latent_loss/args.batch_size,
                    reconst_loss/args.batch_size,
                    (latent_loss+reconst_loss)/args.batch_size, flush=True)
      print('batch', num_batch, time.clock() - begin, flush=True)
      begin = time.clock()
  tes_c,tes_ll,tes_rl = sess.run([test_codes, test_ll, test_rl])
  codes = [ list(c) for c in tes_c ]
  print(tes_ll/test_size,tes_rl/test_size)
  for name, taxon, c in zip(test_read_name, test_taxon, codes):
    print('\t'.join([name, taxon] + list(map(str,c))),file=args.test_codes)
  args.test_codes.close()
