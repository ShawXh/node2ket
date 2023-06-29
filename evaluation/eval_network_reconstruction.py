import argparse
from utils import *

def parse_args():
    '''
    Running for Single-Net embedding.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb1', type=str, default='../baselines/deepwalk/embs/wb-emb.txt', help="/data/emb1.txt")
    parser.add_argument('--emb2', type=str, default='', help="/data/emb2.txt")
    parser.add_argument('--net', type=str, default='/data/xionghao/BTWalk/networks/d2w/wb-net.txt', help="/data/blog-net.txt")
    parser.add_argument('--func', type=str, default='inner_prod', choices=["euc", "cos", "inner_prod"], help="euc")
    parser.add_argument('--miss-info', default=False, action="store_true")
    return parser.parse_args()

args = parse_args()

print("reading emb")
if args.func == "cos":
    norm = True
else:
    norm = False
emb1 = read_emb(args.emb1, norm=norm, given_info=not args.miss_info)
emb2 = read_emb(args.emb2, norm=norm, given_info=not args.miss_info)
print("reading net")
net = read_net(args.net)

if args.func in ["inner_prod", "cos"]:
    print("calculating dist according to inner products")
    # dist = innerproduct_dist(emb1, emb2) 
    dist = innerproduct_dist(emb1, emb1) 
elif args.func == "euc":
    print("calculating dist according to euclidean dist")
    dist = euclidean_dist_2(emb1, emb2)



print("calculating precision")
print("Network Reconstruction Precision:", precision(dist, net))

