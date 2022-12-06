# https://www.python-course.eu/graphs_python.php
# https://networkx.org/documentation/stable/tutorial.html

import networkx as nx
import random
import pandas as pd
import numpy as np


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723 

    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch 
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
  

# função que seleciona os dados e label de cada classificador
def dataset_comite(classificador, loc_x, loc_y, class_comite, model_config):
    
    # if model_config.rede_mlp:
    #    _idloc, _classe = [], []
    #    loc_x = pd.DataFrame(loc_x)
    #    loc_y = pd.DataFrame(loc_y)
    #    for num_classe in dataset_config[class_comite][classificador]:
    #        _idloc.append(loc_y.index[loc_y[0] == num_classe-1].tolist())
    #        _classe = np.concatenate(_idloc)
    #    x_classe = loc_x.iloc[_classe]
    #    y_classe = loc_y.iloc[_classe]
    #    return pd.DataFrame(x_classe), pd.DataFrame(y_classe)
    # elif model_config.rede_lstm:
    indices = np.concatenate([np.where(loc_y==x-1)[0].tolist() for x in class_comite[classificador]], axis=0)
    indices= [int(i) for i in indices]
    indices = np.array(indices)
    
    return loc_x[indices], loc_y[indices]

# função utiliza a clafficador e o label e retorna em que saída o label vai se tornar no classificador
def mapeamento_classe(comite, classificador, label, hier_label):
  
  ind = 0
  if len(comite[classificador]) > 1:
    for no in dataset_config[hier_label][classificador].values():
      if (label+1) in no:
        return ind
      ind = ind+1
  else:
    for no in dataset_config[hier_label][classificador].values():
      for xx in no:
        if (label+1) == xx:
          return ind
        ind  = ind + 1
  return None

def changelabel_comite(lista, label, nos):
    if nos > 1:
        return [lista.index(x) for x in lista if label in x]
    else:
        return [x.index(label) for x in lista if label in x]
    # return [(lista.index(x), x.index(label)) for x in lista if label in x]
    
