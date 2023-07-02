import networkx as nx
import argparse
import numpy as np
from copy import deepcopy
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str)
    parser.add_argument('--ratio', type = float, default=0.01, help="the ratio of masked edges")
    return parser.parse_args()

args = parse_args()

filename = "./ca-GrQc-net.txt"
filename = "./PPI/PPI-net.txt"
filename = "./blog/blog-net.txt"
filename = "./dblp-net.txt"
filename = args.net

g = nx.read_edgelist(filename, data=(('weight',float),), delimiter=" ", nodetype=int).to_undirected()
N = g.number_of_nodes()
M = g.number_of_edges()
edges = list(g.edges())
print("Initial edges:", M)

mask_ratio = 0.01
mask_ratio = args.ratio
noise_ratio = 1

# generate masked edges
masked_g = deepcopy(g).to_undirected()
masked_edges = []
last_time = time.time()
while (len(masked_edges) < mask_ratio * M and time.time() - last_time < 10):
    edge_idx = np.random.choice(np.arange(masked_g.number_of_edges()), 1)[0]
    i, j = list(masked_g.edges())[edge_idx]
    if (masked_g.degree(i) >= 2 and masked_g.degree(j) >= 2):
        masked_edges.append((i, j))
        masked_g.remove_edge(i, j)
        print("Progress [%d/%d]" % (len(masked_edges), mask_ratio * M))
        last_time = time.time()
print("Masked edges:", len(masked_edges))

with open("./link_pred_data/" + filename.split("/")[-1] + "-masked-edges", "w") as f:
    for (i, j) in masked_edges:
        f.write("%d %d 1\n" % (i, j))

# generate noise edges
noise_edges = []
noise_g = nx.Graph()
while (len(noise_edges) < noise_ratio * len(masked_edges)):
    i = np.random.randint(N)
    j = np.random.randint(N)
    if (i != j and not g.has_edge(i, j) and not noise_g.has_edge(i, j)):
        noise_edges.append((i, j))
        noise_g.add_edge(i, j)
print("Noise edges:", len(noise_edges))

with open("./link_pred_data/" + filename.split("/")[-1] + "-noise-edges", "w") as f:
    for (i, j) in noise_edges:
        f.write("%d %d 1\n" % (i, j))

# saving masked networks
print("After removing edges:", masked_g.number_of_edges())
for n in masked_g.nodes():
    if masked_g.degree[n] == 0:
        print("Warning: has zero-degree nodes", n)

with open("./link_pred_data/" + filename.split("/")[-1] + "-masked", "w") as f:
    for (i, j) in masked_g.edges():
        f.write("%d %d 1\n" % (i, j))
        f.write("%d %d 1\n" % (j, i))


