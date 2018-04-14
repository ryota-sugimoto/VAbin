import matplotlib
import matplotlib.cm as cm
matplotlib.use('Agg')
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('code_file', type=argparse.FileType('r'))
args = parser.parse_args()

taxons = []
codes = []
for s in args.code_file:
  l = s.strip().split()
  taxons.append(l[1])
  codes.append([float(n) for n in l[2:]])

n_taxons = len(set(taxons))
sorted_taxons = sorted(list(set(taxons)))
print(sorted_taxons)
numbering_taxons = {taxon:i for i,taxon in enumerate(sorted_taxons)}
numbered_taxons = [numbering_taxons[taxon] for taxon in taxons]

'''
bacterias = {'E.coli', 'A.h', 'A.s'}
virals = {'T4', 'RB43', 'RB49', 'RB69', 'Aeh1', '25', '31', '44RR2.8t'}
'''

colors = cm.jet(np.linspace(0,1,n_taxons))
data = { taxon: {'color': colors[n],
                 'x': [],
                 'y': [],
                 'z': [] } for n,taxon in enumerate(sorted_taxons) } 

pca = PCA(n_components=3)
embed = pca.fit_transform(preprocessing.scale(codes))
for x,y,z,taxon in zip(embed[:,0], embed[:,1], embed[:,2], taxons):
  data[taxon]['x'].append(x)
  data[taxon]['y'].append(y)
  data[taxon]['z'].append(z)

fig1 = plt.subplot(111)
handles = []
for taxon in sorted(data.keys()):
  handles.append(fig1.scatter(data[taxon]['x'], data[taxon]['y'],
                              linewidths=0,
                              alpha=0.5,
                              c=data[taxon]['color'],
                              label=taxon))
fig1.legend(handles=handles)
plt.savefig(args.code_file.name.split('/')[-1]+'.pca.png')

'''
for x,y,taxon,c in zip(embed[:,0], embed[:,1], taxons, colors):
  if taxon in bacterias:
    scatter_handles.append(
      plt.scatter(x, y, s=0.5, linewidths=0, c=c, marker='o',label=taxon))
  else:
    scatter_handles.append(
      plt.scatter(x, y, s=0.5, c=c, marker='x',label=taxon))
fig_handle.legend(handles=scatter_handles)
plt.savefig('pca.png',dpi=800)
'''
