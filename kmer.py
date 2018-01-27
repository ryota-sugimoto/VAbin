#!/usr/bin/env python
from collections import defaultdict
from itertools import product

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
    seq = seq.upper()
    count = defaultdict(int)
    for i in range(len(seq)-self.k+1):
      count[seq[i:i+self.k]] += 1
    count = self.balance_strand(count)
    v = []
    for mer in self.balanced_kmer:
      v.append(count[mer])
    return v
       
if __name__ == '__main__':
 import argparse
 parser = argparse.ArgumentParser()
 parser.add_argument('k', type=int)
 parser.add_argument('file', type=argparse.FileType('r'))
 args = parser.parse_args()
 
 counter = KmerCounter(args.k)
 
'''
 for s in args.file:
   contig_name = s.strip()[1:]
   contig = next(args.file).strip()
   kmer_count = counter.count(contig)
   print('\t'.join([contig_name]+list(map(str,kmer_count))))
'''

for s in args.file:
  readname=s.strip()[1:]
  seq = next(args.file).strip().upper()
  kmer_count = counter.count(seq)
  print('\t'.join([readname]+list(map(str,kmer_count))))
  next(args.file),next(args.file)
