import networkx as nx
import numpy as np

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str)
    parser.add_argument('--n_walks', type = int, default=1, help="number of walks for each node")
    parser.add_argument('--walk_length', type = int, default=80, help="the length of each random walk sequence")
    return parser.parse_args()

args = parse_args()

net_path = args.net

n_walks = args.n_walks
walk_length = args.walk_length

g = nx.read_edgelist(net_path, data=(('weight',float),), delimiter=" ", nodetype=int)
g = g.to_undirected()

# with open(net_path + ".rw", "w") as f:
with open(net_path + ".rw_nwalks_%d_length_%d" % (n_walks, walk_length), "w") as f:
    for w in range(n_walks):
        print(w)
        for node in list(g.nodes()):
            walk = [node]
            while len(walk) < walk_length:
                node = int(np.random.choice(list(nx.all_neighbors(g, node))))
                walk.append(node)
            walk = list(map(str, walk))
            f.write(" ".join(walk) + "\n")
