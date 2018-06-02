#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import time
import itertools

from multimer_model import KmerAutoencoder

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('train_kmer_file', type=argparse.FileType('r'))
parser.add_argument('test_kmer_file', type=argparse.FileType('r'))
parser.add_argument('test_codes_out', type=str)
parser.add_argument('--num_codes', type=int, default=10)
parser.add_argument('--num_kmer_hidden_layers', type=int, default=8)
parser.add_argument('--num_hidden_layers', type=int, default=8)
parser.add_argument('--num_neurons', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--max_batch', type=int, default=100000)
parser.add_argument('--report_per_batch', type=int, default=10)
args = parser.parse_args()

def data_generator(f):
  read_length = sum(map(float, next(f).split()[1:3]))
  f.seek(0)
  kmer_sizes = [2, 10, 32, 136, 512, 2080]
  sizes = [ [read_length-n]*size for n,size in enumerate(kmer_sizes) ]
  sizes = np.array(list(itertools.chain.from_iterable(sizes)), dtype=np.float32)
  def gen():
    while True:
      for s in f:
        r = np.array([float(n) for n in s.strip().split()[1:]],
                      dtype=np.float32)
        yield r
      f.seek(0)
  return gen
dataset = tf.data.Dataset.from_generator(data_generator(args.train_kmer_file),
                                         tf.float32,
                                         [2772])
batch = dataset.batch(batch_size=args.batch_size)
batch_iterator = batch.make_one_shot_iterator()

test_kmer = []
test_read_name = []
test_taxon = []
read_length = sum(map(float, next(args.test_kmer_file).split()[2:4]))
args.test_kmer_file.seek(0)
kmer_sizes = [2, 10, 32, 136, 512, 2080]
sizes = [ [read_length-n]*size for n,size in enumerate(kmer_sizes) ]
sizes = np.array(list(itertools.chain.from_iterable(sizes)), dtype=np.float32)
for s in args.test_kmer_file:
  l = s.strip().split()
  test_read_name.append(l[0])
  test_taxon.append(l[1])
  r = np.array([ float(n) for n in l[2:]], dtype=np.float32)
  test_kmer.append(r)
test_kmer = np.array(test_kmer)
test_size = len(test_read_name)

KA = KmerAutoencoder(num_neurons=args.num_neurons,
                     num_hidden_layers=args.num_hidden_layers,
                     num_codes=args.num_codes,
                     num_kmer_hidden_layers=args.num_kmer_hidden_layers)
ll,rl,opt = KA.train(batch_iterator.get_next(), learning_rate=0.00001)

_,_,test_codes,_,_,test_ll,test_rl = KA.model(test_kmer)

init = tf.global_variables_initializer()
num_batch = 0
with tf.Session() as sess:
  init.run()
  begin = time.clock()
  while num_batch < args.max_batch:
    try:
      latent_loss, reconst_loss, _ = sess.run([ll,rl,opt])
    except tf.errors.OutOfRangeError:
      break
    num_batch += 1
    if num_batch % args.report_per_batch == 0:
      print('batch', num_batch, time.clock() - begin, flush=True)
      print('train_loss', latent_loss/args.batch_size,
                    reconst_loss/args.batch_size,
                    (latent_loss+reconst_loss)/args.batch_size, flush=True)
      begin = time.clock()
      
      tes_c,tes_ll,tes_rl = sess.run([test_codes, test_ll, test_rl])
      codes = [ list(c) for c in tes_c ]
      print('test_loss', tes_ll/test_size, tes_rl/test_size,
                         (tes_ll+tes_rl)/test_size, flush=True)
      code_file = open(args.test_codes_out+'_'+str(num_batch), 'w')
      for name, taxon, c in zip(test_read_name, test_taxon, codes):
        print('\t'.join([name, taxon] + list(map(str,c))),file=code_file)
      code_file.close()
  
  tes_c,tes_ll,tes_rl = sess.run([test_codes, test_ll, test_rl])
  codes = [ list(c) for c in tes_c ]
  print('test_loss', tes_ll/test_size, tes_rl/test_size)
  code_file = open(args.test_codes_out+'_'+str(num_batch), 'w')
  for name, taxon, c in zip(test_read_name, test_taxon, codes):
    print('\t'.join([name, taxon] + list(map(str,c))),file=code_file)
  code_file.close()
