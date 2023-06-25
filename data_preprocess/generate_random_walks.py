import networkx as nx
import numpy as np

net_path = ""

n_walks = 10
n_walks = 1 # for less data
walk_length = 80

g = nx.read_edgelist(net_path, data=(('weight',float),), delimiter=" ", nodetype=int)
g = g.to_undirected()

# with open(net_path + ".rw", "w") as f:
with open(net_path + ".rw_nwalks_1", "w") as f:
    for w in range(n_walks):
        print(w)
        for node in list(g.nodes()):
            walk = [node]
            while len(walk) < walk_length:
                node = int(np.random.choice(list(nx.all_neighbors(g, node))))
                walk.append(node)
            walk = list(map(str, walk))
            f.write(" ".join(walk) + "\n")
