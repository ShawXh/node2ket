import networkx as nx
import argparse
from collections import defaultdict

# resolutions = [1000, 500, 100]
resolutions = [1000, 800]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', type=int, default=2)
    parser.add_argument('--res', type=int, nargs="+", default=[], help="resolution")
    parser.add_argument('-L', type=int, default=1)
    parser.add_argument('--net', type=str, default='')
    return parser.parse_args()

args = parse_args()
resolutions = list(map(int, args.res))
print("resolutions:", resolutions)

g = nx.read_edgelist(args.net, data=(('weight',float),), delimiter=" ", nodetype=int)

TUdict = {}
for node in range(g.number_of_nodes()):
    TUdict[node] = {}
    TUdict[node][0] = node

for c in range(min(args.C, len(resolutions))):
    it = nx.community.louvain_communities(g, resolution=resolutions[c], seed=c)
    for r, community in enumerate(it):
        for node in community:
            TUdict[node][c] = r
if len(resolutions) < args.C:
    for c in range(len(resolutions), args.C):
        for node in g.nodes():
            TUdict[node][c] = node

all_R = 0
R = []
for c in range(args.C):
    maxr = 0
    for node in g.nodes():
        maxr = max(TUdict[node][c], maxr)
    R.append(str(maxr + 1))
    all_R += maxr + 1
print("Sum of rows:", all_R)


with open(args.net.split("/")[-1] + ".louvain_config", "w") as f:
    f.write("%d %d 1\n" % (args.L, args.C))
    f.write(" ".join(R) + "\n")
    for node in TUdict:
        line = str(node) + " " + " ".join([str(TUdict[node][c]) for c in range(args.C)]) + "\n"
        f.write(line)
