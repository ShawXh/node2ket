import numpy as np
from collections import defaultdict

def read_emb(file_name, norm=True, given_info=True):
    error_line_flag = False
    if given_info:
        with open(file_name, "r") as f:
            ln = f.readline()
            n, d = list(map(int, ln.strip().split(" ")))
            print("%d nodes, %d dim" % (n, d))
            emb = np.zeros((n, d))
            for ln in f.readlines():
                if ln.startswith("</s>"): 
                    error_line_flag = True
                    continue
                ln = ln.strip().split(" ")
                nd = int(ln[0])
                emb[nd] = np.array(list(map(float, ln[1:])))
        if error_line_flag:
            emb = emb[:n-1]
    else:
        embd = {}
        with open(file_name, "r") as f:
            for ln in f.readlines():
                if ln.startswith("</s>"): 
                    error_line_flag = True
                    continue
                ln = ln.strip().split(" ")
                nd = int(ln[0])
                embd[nd] = np.array(list(map(float, ln[1:])))
        n = max(embd.keys()) + 1
        assert n == len(embd.keys()), "embedding file is not complete!"
        d = embd[list(embd.keys())[0]].shape[0]
        print("Information missed, detected %d nodes, %d dim" % (n, d))
        emb = np.zeros((n, d))
        for node in embd:
            emb[node] = embd[node]
        if error_line_flag:
            emb = emb[:n-1]
    if norm:
        m = np.sqrt(np.sum(emb * emb, 1).reshape(n, 1))
        return emb / m
    else:
        return emb

def read_net(file_name):
    net = defaultdict(dict)
    with open(file_name, "r") as f:
        for ln in f.readlines():
            ln = ln.strip().split(" ")
            n1, n2 = list(map(int, ln))[:2]
            net[n1][n2] = 1
            net[n2][n1] = 1
    return net

def euclidean_dist_1(emb1, emb2):
    N = emb1.shape[0]
    dist = np.zeros((N, N))
    for nd in range(N):
        emb = emb1[nd]
        d = emb2 - emb
        d = np.sum(d * d, axis=1)
        dist[nd] = d
    return dist

def euclidean_dist_2(emb1, emb2):
    N1 = np.sum(emb1 * emb1, axis=1)
    NN = emb1.dot(emb2.transpose())
    N2 = np.sum(emb2 * emb2, axis=1)
    dist = -2 * NN + np.expand_dims(N1, 1) + N2
    for i in range(emb1.shape[0]):
        dist[i, i] = 999999
    return dist

def innerproduct_dist(emb1, emb2):
    dist = -emb1.dot(emb2.transpose())
    for i in range(emb1.shape[0]):
        dist[i, i] = 999999
    return dist

def precision(dist, net):
    N = dist.shape[0]
    dist = dist.reshape(N * N)
    rank = np.argsort(dist)
    M = 0
    for nd in net:
        M += len(net[nd])
    rank = rank[:M]

    cnt = 0
    for i in range(M):
        a = int(rank[i] / N)
        b = int(rank[i] % N)
        if b in net[a]:
            cnt += 1
    return cnt / M
