# node2ket

This repo contains the code and data for the submission of node2ket to VLDB2024. The files include:

```
./
├── CMakeLists.txt
├── compile.sh
├── data
│   ├── ca-GrQc-net.txt (for network reconstruction and node classification)
│   ├── ca-GrQc-net.txt-masked (for link prediction)
│   ├── ca-GrQc-net.txt-masked-edges (for link prediction)
│   ├── ca-GrQc-net.txt-noise-edges (for link prediction)
│   └── data_node2ket.zip (other datasets)
├── data_preprocess
│   ├── generate_random_walks.py (for node2ket with sequences as the input)
│   ├── link_pred_process.py (preprocess data for link prediction)
│   └── README.md
├── evaluation (code for evaluation over node2ket and any other baselines)
│   ├── EmbLoader.py
│   ├── eval_link_pred.py
│   ├── eval_network_reconstruction.py
│   ├── run-nr.sh
│   ├── utils.py
│   └── EvalNR (excutable file after compilation)
├── InitTUConfigLouvain.py (generate sub-embedding indices for node2ket+ by Louvain partition)
├── node2ket (excutable file after compilation)
├── node_embedding.txt (node embeddings by node2ket)
├── sub_embedding.txt (sub embeddings by node2ket)
├── src (cpp source code)
│   ├── EvalNR.cpp
│   ├── libn2k
│   │   ├── emb.cpp
│   │   ├── emb.h
│   │   ├── net.cpp
│   │   └── net.h
│   └── node2ket.cpp
├── README.md
├── run-ablation.sh
├── run-eval_params.sh
├── run-exp.sh
└── run-parallel.sh
```

## Requirements

The python scripts work on python 3.8.13 with networkx==2.8.4 and numpy==1.22.4.


# How to Compile

```
sh compile.sh
```

# How to Run node2ket

## Running Example

**Step 1.** Prepare the data, i.e. the network file, which are in the form of weigted edgelist:
```
node_id node_id weight
...
```

**Step 2 (necessary for node2ket+).** Build index table for sub-embeddings. An example:
```
python InitTUConfigLouvain.py --net ./data/ca-GrQc-net.txt -L 1 -C 8 --res  100 500 1000 1500
```

**Step 3.** Run the program. Examples:

node2ket:
```
./node2ket -net ./data/ca-GrQc-net.txt -dim 16 -C 8 -rw 1 -window-size 2 -obj mt -samples 100 -print 1 -eval-nr 1 -rho 0.1
```

node2ket+:
```
./node2ket -net ./data/ca-GrQc-net.txt -dim 16 -C 8 -rw 1 -window-size 2 -obj mt -samples 100 -riemann 0 -print 1 -eval-nr 1 -rho 0.1 -config ca-GrQc-net.txt.louvain_config -thread 8
```

## Arguments of Node2ket

Input data:
- -net (str) Path of the input network file.
- -seq (str) Path of the input node sequence file. 
- -config (str, for node2ket+) Path of the index table of sub-embeddings.

Embedding dimensions:
- -C (int) The number of sub-embedding for each node.
- -dim (int) The dimension of sub-embeddings.

Objetives:
- -obj (str) The objective. Can be set as either mt or sgns.
- -mt-mar (float) The margin of the loss marginal triplet
- -num-neg (int) The number of negative samples for the loss skip-gram by negative sampling.

Sampling strategies:
- -rw (int) If set as 1, then use random walk as the sampling strategy.
- -window-size (int) The window size of random walk.
- -rwr (int) If set as 1, then use random walk with restart as the sampling strategy.
- -ppralpha (float) The probability of restarting in random walk with restart.

Optimizer:
- -opt (str) The optimizer. Can be set as sgd, bsgd, rmsprop, or adagrad. Default is adagrad.
- -batch-size (int) The batch size.
- -riemann (int) The order in Riemannian optimization.
- -rho (float) Learning rate.

Verbose print:
- -print (int) Set to 1 to print training details.

Evaluation:
- -eval-nr (int) Set to 1 to evaluate the network reconstruction precision after embedding learning.

Output:
- -outputemb (int) Set to 1 to output embeddings.
- -node-emb (str) Path of output full-dimensional node embeddings.
- -sub-emb (str) Path of output sub-embeddings, from which the full-dimensional embeddings can be recovered.

# Experiments

## node2ket

All the scripts of experiments in the paper are given as run_*.sh files.

## Baselines

Supposing we have an embedding file named as emb.txt, whose format is as follows:
```
num_nodes emb_dim
node_id -0.007614 0.142711 0.229157 -0.013976 0.196722 -0.156228 -0.915828 6.143982 .. (embedding vectors)
...
```
Then the node embeddings are evaluated by following scripts:

**Network Reconstruction:**

```
python network_reconstruction.py --emb1 emb.txt --emb2 emb.txt --net /data/ca-GrQc-net.txt --func euc
```

**Link Prediction:**
```
python eval_link_pred.py --net /data/ca-GrQc-net.txt-masked -c --conv-emb emb.txt
```

**Node Classification:**

See this [repo](https://github.com/ShawXh/Evaluate-Embedding).