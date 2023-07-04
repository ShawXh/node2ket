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
python InitTUConfigLouvain.py --net ./data/ca-GrQc-net.txt -C 8 --res  100 500 1000 1500
```

**Step 3.** Run the program. Examples:

node2ket:
```
./node2ket -net ./data/ca-GrQc-net.txt -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 1 -eval-nr 1 -rho 0.1
```

node2ket+:
```
./node2ket -net ./data/ca-GrQc-net.txt -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -riemann 0 -print 1 -eval-nr 1 -rho 0.1 -config ca-GrQc-net.txt.louvain_config -thread 8
```

## Public Arguments of Node2ket

Input data:
- -net (str) Path of the input network file.
- -seq (str) Path of the input node sequence file. 
- -config (str, for node2ket+) Path of the index table of sub-embeddings.

Embedding dimension:
- -C (int) The number of sub-embedding for each node.
- -dim (int) The dimension of sub-embeddings.

Objetive:
- -obj (str) The objective. Can be set as either mt or sgns.
- -mt-mar (float) The margin (gamma in the paper) of the loss marginal triplet.
- -num-neg (int) The number of negative samples for the loss skip-gram by negative sampling.

Sampling strategy:
- -rw (0 or 1) If set as 1, then use random walk as the sampling strategy.
- -window-size (int) The window size of random walk.
- -rwr (0 or 1) If set as 1, then use random walk with restart as the sampling strategy.
- -ppralpha (float) The probability of restarting in random walk with restart.

Optimizer:
- -opt (str) The optimizer. Can be set as sgd, bsgd, rmsprop, or adagrad. Default is adagrad.
- -batch-size (int) The batch size.
- -riemann (int) The order in Riemannian optimization. Can be set as 0, 1, or 2.
- -rho (float) Learning rate.

Constraints:
- -zero (0 or 1) Set to 1 to adopt zero constraints. Default is 1.
- -norm (0 or 1) Set to 1 to normalize the sub embeddings on the unit hyper sphere. Default is 1.

Verbose print:
- -print (0 or 1) Set to 1 to print training progress.

Evaluation:
- -eval-nr (0 or 1) Set to 1 to evaluate the network reconstruction precision after embedding learning.

Output:
- -outputemb (0 or 1) Set to 1 to output embeddings.
- -node-emb (str) Path of output full-dimensional node embeddings. Default is ./node_embedding.txt.
- -sub-emb (str) Path of output sub-embeddings, from which the full-dimensional embeddings can be recovered. Default is ./sub_embedding.txt.

Others:
- -seed (int) The random seed.
- -thread (int) The nubmer of CPU threads.
- -samples (int) The number of running iterations (in million). Set to 1 means running for 1 million iterations.


## Generate Sub-Embedding Indices by Louvain Partition for Node2ket+

Example:
```
python InitTUConfigLouvain.py --net ./data/ca-GrQc-net.txt -C 8 --res 100 500 1000 1500
```

The argument "-C" is the number of sub-embeddings for each node, "--res" is resolution by Louvain Partition. In the example above, "--res 100 500 1000 1500" means that it conduct Louvain Parition for 4 times with resolutions 100, 500, 1000, and 1500 to generate sub-embedding indices for 4 of the 8 sub-embeddings, and left un-partitioned for the rest 4 of the 8 sub-embeddings. The results of sub-embedding indices are stored in the file "ca-GrQc-net.txt.louvain_config".


# Experiments

All the scripts to reproduce experiments of node2ket and node2ket+ in the paper are given as run_*.sh files (maybe annotated).

## Evaluate node2ket

**Network Reconstruction:**

For middle-scale networks, run node2ket with the option "-eval-nr 1".

For large-scale networks /data/youtube-idxnorm-net.txt where the computing the full adjacency matrix is impractical:

```
cd evaluation
node2ket: ./EvalNR -emb ./sub_embedding.txt -tensorized 1 -net /data/youtube-idxnorm-net.txt
node2ket+: ./EvalNR -emb ./sub_embedding.txt -tensorized 1 -net /data/youtube-idxnorm-net.txt -config youtube-idxnorm-net.txt.louvain_config
```

**Link Prediction:**

```
cd evaluation
python eval_link_pred.py --net /data/ca-GrQc-net.txt-masked -t --tu-emb ./sub_embedding.txt
```


**Node Classification:**

Following instructions in this [repo](https://github.com/ShawXh/Evaluate-Embedding) with the full-dimensional node embedding file as input.

## Evaluate Arbitrary Baselines

Supposing we have a full-dimensional embedding file named as emb.txt, whose format is as follows:
```
num_nodes emb_dim
node_id -0.007614 0.142711 0.229157 -0.013976 0.196722 -0.156228 -0.915828 6.143982 .. (embedding vectors)
...
```
Then the node embeddings are evaluated by following scripts:

**Network Reconstruction:**

On middle-scale networks:

```
cd evaluation
python eval_network_reconstruction.py --emb1 emb.txt --emb2 emb.txt --net /data/ca-GrQc-net.txt --func euc
```

On large-scale networks where the computing the full adjacency matrix is impractical:

```
cd evaluation
./EvalNR -emb emb.txt -net /data/youtube/youtube-idxnorm-net.txt -tensorized 0
```

**Link Prediction:**
```
cd evaluation
python eval_link_pred.py --net /data/ca-GrQc-net.txt-masked -c --conv-emb emb.txt
```

**Node Classification:**

Following this [repo](https://github.com/ShawXh/Evaluate-Embedding).