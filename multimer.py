#!/usr/bin/env python
from collections import defaultdict
from itertools import product
from itertools import chain

def reverse_complement(seq):
  comp = {'A': 'T',
          'C': 'G',
          'G': 'C',
          'T': 'A'}
  seq = list(seq)
  seq.reverse()
  return ''.join(map(lambda n:comp[n], seq))

class KmerCounter():
  def __init__(self, k):
    self.k = k
    self._calc_kmer()

  def _calc_kmer(self):
    self.kmer = [''.join(mer) for mer in product('ACGT', repeat=self.k)]
    self.kmer.sort()
    pass_set = set([])
    self.balanced_kmer = []
    for mer in self.kmer:
      comp = reverse_complement(mer)
      if mer not in pass_set:
        self.balanced_kmer.append(mer)
      pass_set.add(comp)
      pass_set.add(mer)
    self.balanced_kmer.sort()
 
  def balance_strand(self, count):
    for mer in self.balanced_kmer:
      comp = reverse_complement(mer)
      if mer != comp:
        new_value = count[mer] + count[comp]
        count[mer] = count[comp] = new_value
    return count

  def count(self, seq):
    seq = seq.strip().upper()
    length = len(seq)
    count = defaultdict(int)
    for i in range(len(seq)-self.k+1):
      count[seq[i:i+self.k]] += 1.0
    count = self.balance_strand(count)
    v = []
    for mer in self.balanced_kmer:
      v.append(count[mer]/length)
    return v
       
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_k', default=6, type=int)
  parser.add_argument('--min_read_length', default=1000, type=int)
  parser.add_argument('fastq1', type=argparse.FileType('r'))
  parser.add_argument('fastq2', type=argparse.FileType('r'), nargs='?')
  args = parser.parse_args()
  
  counters = [KmerCounter(k) for k in range(1, args.max_k+1)]
 
  for s in args.fastq1:
    readname=s.strip()[1:].split('#')[0]
    seq1 = next(args.fastq1).strip().upper()
    if args.fastq2:
      next(args.fastq2)
      seq2 = next(args.fastq2).strip().upper()
    if len(seq1) >= args.min_read_length:
      count1 = [counter.count(seq1) for counter in counters]
      count1 = chain.from_iterable(count1)
      if args.fastq2:
        count2 = [counter.count(seq2) for counter in counters]
        count2 = chain.from_iterable(count2)
        kmer_count = map(lambda x,y:x+y, count1, count2)
        print('\t'.join([readname]+list(map(str,kmer_count))))
      else:
        print('\t'.join([readname]+list(map(str,count1))))
    next(args.fastq1), next(args.fastq1)
    if args.fastq2:
      next(args.fastq2), next(args.fastq2)
