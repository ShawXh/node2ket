import numpy as np
import networkx as nx

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

class ConventionalEmb(object):
    def __init__(self, emb_file):
        self.emb = self._read_emb(emb_file, norm=False)
        self.dim = self.emb.shape[1]

    def _read_emb(self, file_name, norm=True, given_info=True):
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

    def _read_emb_v0(self, file_name, norm=True, given_info=True):
        if given_info:
            with open(file_name, "r") as f:
                ln = f.readline()
                n, d = list(map(int, ln.strip().split(" ")))
                print("From embedding file: %d nodes, %d dim" % (n, d))
                emb = np.zeros((n, d))
                for ln in f.readlines():
                    ln = ln.strip().split(" ")
                    nd = int(ln[0])
                    emb[nd] = np.array(list(map(float, ln[1:])))
        else:
            embd = {}
            with open(file_name, "r") as f:
                for ln in f.readlines():
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
        if norm:
            m = np.sqrt(np.sum(emb * emb, 1).reshape(n, 1))
            return emb / m
        else:
            return emb

    def get_euclidean_score(self, i, j):
        delta = self.emb[i] - self.emb[j]
        return -delta.dot(delta)

class TUEmb(object):
    def __init__(self, tu_emb_file, tu_config_file=None):
        tu_emb_dict = self._read_emb(tu_emb_file)
        self.L = tu_emb_dict["L"]
        self.C = tu_emb_dict["C"]
        self.R = tu_emb_dict["R"]
        self.dim = tu_emb_dict["dim"]
        self.tu_emb = tu_emb_dict["TU_emb"]
        if tu_config_file is not None:
            self.tu_config = self._read_tu_config(tu_config_file)
        else:
            self.tu_config = None
    
    def _read_emb(self, tu_emb_file):
        '''
        return
        ------
        tu_dict dict: 
            tu_dict = {
                "L": L, # int
                "C": C, # int
                "R": R, # int/list
                "dim": dim, # int
                "TU_emb": tu_emb 
                # dict of np.array, tu_emb[l][c].shape = (R, dim) is the embedding matrix of Layer-l Column-c TU
                # embeddings of the Tensorized Embedding Block
            }
        '''
        tu_emb = {}
        with open(tu_emb_file, "r") as f:
            ln = f.readline()
            ln = ln.strip().split(" ")
            if len(ln) == 4:
                L = int(ln[0].split("=")[-1])
                C = int(ln[1].split("=")[-1])
                R = int(ln[2].split("=")[-1])
                dim = int(ln[3].split("=")[-1])
            if len(ln) == 3:
                L = int(ln[0].split("=")[-1])
                C = int(ln[1].split("=")[-1])
                dim = int(ln[2].split("=")[-1])
                ln = f.readline()
                R = list(map(int, ln.strip().split(" ")[1:]))
            if len(ln) > 4:
                L = int(ln[0].split("=")[-1])
                C = int(ln[1].split("=")[-1])
                R = [int(ln[2].split("=")[-1])]
                for _ in range(C - 1):
                    R.append(int(ln[3 + _]))
                dim = int(ln[2 + C].split("=")[-1])
            print("L=", L)
            print("C=", C) 
            print("R=", R)
            print("d=", dim)
            for l in range(L): 
                tu_emb[l] = {}
                for c in range(C):
                    if isinstance(R, list):
                        tu_emb[l][c] = np.empty((R[c], dim))
                    if isinstance(R, int):
                        tu_emb[l][c] = np.empty((R, dim))
            for ln in f.readlines():
                if ln.startswith("Row"): break
                ln = ln.strip().split(" ")
                tu_emb[int(ln[0])][int(ln[1])][int(ln[2])] = np.array(list(map(float, ln[3:])))
            tu_dict = {
                "L": L,
                "C": C,
                "R": R,
                "dim": dim,
                "TU_emb": tu_emb
            }
        return tu_dict

    def _read_tu_config(self, tu_config_file):
        with open(tu_config_file, "r") as f:
            ln = f.readline()
            ln = list(map(int, ln.strip().split(" ")))
            L, C, simple_config = ln # simple config is left unused
            assert L == self.L, "TU_configs (L=%d) conlficts with TU_embeddings (L=%d)" % (L, self.L)
            assert C == self.C, "TU_configs (C=%d) conlficts with TU_embeddings (C=%d)" % (L, self.C)
            tu_config = {}
            for ln in f.readlines()[1:]:
                ln = list(map(int, ln.strip().split(" ")))
                tu_config[ln[0]] = ln[1:]
        return tu_config

    def _get_row(self, i, c):
        if self.tu_config is None:
            return i
        else:
            return self.tu_config[i][c]

    def get_inner_prod_score(self, i ,j):
        '''
        i, j int: the node idx
        '''
        all_ip = 0.
        for ll in range(self.L * self.L):
            l1 = ll // self.L
            l2 = ll % self.L
            layer_ip = 1.
            for c in range(self.C):
                layer_ip *= self.tu_emb[l1][c][self._get_row(i, c)].dot(self.tu_emb[l2][c][self._get_row(j, c)])
            all_ip += layer_ip
        return all_ip

    def get_sigmoid_score(self, i, j):
        return sigmoid(self.get_inner_prod_score(i, j))

class Network(object):
    def __init__(self, edgelist_file):
        self.g = nx.read_edgelist(edgelist_file, data=(('weight',float),), delimiter=" ", nodetype=int)
        self.g = self.g.to_undirected()

    def _get_all_node_tu_emb(self):
        self.node_tuemb = np.zeros((self.g.number_of_nodes(), self.tuemb.L, self.tuemb.C, self.tuemb.dim))
        for l in range(self.tuemb.L):
            for n in range(self.g.number_of_nodes()):
                for c in range(self.tuemb.C):
                    self.node_tuemb[n][l][c] = self.tuemb.tu_emb[l][c][self.tuemb._get_row(n, c)]

    def load_TU_emb(self, tu_emb_file, tu_config_file=None):
        print("Loading TU embeddings from", tu_emb_file, "and", str(tu_config_file))
        self.tuemb = TUEmb(tu_emb_file, tu_config_file)
        self._get_all_node_tu_emb()

    def load_conventional_emb(self, emb_file):
        print("Loading conventional embeddings from", emb_file)
        self.convemb = ConventionalEmb(emb_file)

    def task_link_pred(self, masked_link_file, noise_link_file, conv_emb_file=None, tu_emb_file=None, tu_config_file=None):
        self.masked_g = nx.read_edgelist(masked_link_file, data=(('weight',float),), delimiter=" ", nodetype=int).to_undirected()
        print("Edges in masked_g:", self.masked_g.number_of_edges())
        self.noise_g = nx.read_edgelist(noise_link_file, data=(('weight',float),), delimiter=" ", nodetype=int).to_undirected()
        print("Edges in noise_g:", self.noise_g.number_of_edges())
        if conv_emb_file is not None:
            self.load_conventional_emb(conv_emb_file)
            masked_score = [self.convemb.get_euclidean_score(i, j) for (i, j) in self.masked_g.edges()]
            noise_score = [self.convemb.get_euclidean_score(i, j) for (i, j) in self.noise_g.edges()]
        if tu_emb_file is not None:
            self.load_TU_emb(tu_emb_file, tu_config_file)
            masked_score = [self.tuemb.get_inner_prod_score(i, j) for (i, j) in self.masked_g.edges()]
            noise_score = [self.tuemb.get_inner_prod_score(i, j) for (i, j) in self.noise_g.edges()]
        scores = np.array(masked_score + noise_score)
        scores = np.sort(scores)[::-1]
        threshold = scores[len(masked_score)]
        print("Num scored edges", len(scores))
        print("Max score", scores[0], ", threshold", threshold, ", min score", scores[-1])
        # print(np.array(masked_score))
        precision = np.sum(np.array(masked_score) > threshold) / len(masked_score)
        print("Link predict precision ", precision)
        return

    def task_network_reconstruct(self, conv_emb_file=None, tu_emb_file=None, tu_config_file=None):
        pass

    def task_node_classification(self, label_file, conv_emb_file=None, tu_emb_file=None, tu_config_file=None):
        # TODO
        if conv_emb_file is not None:
            self.load_conventional_emb(conv_emb_file)
        if tu_emb_file is not None:
            self.load_TU_emb(tu_emb_file, tu_config_file)