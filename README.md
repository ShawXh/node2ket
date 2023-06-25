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
│   └── ca-GrQc-net.txt-noise-edges (for link prediction)
├── data_preprocess
│   ├── generate_random_walks.py (for node2ket with sequences as the input)
│   └── link_pred_process.py (preprocess data for link prediction)
├── evaluation (code for evaluation over node2ket and any other baselines)
│   ├── EmbLoader.py
│   ├── eval_link_pred.py
│   ├── eval_network_reconstruction
│   │   ├── network_reconstruction.py
│   │   ├── run-nr.sh
│   │   └── utils.py
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

The python scripts work on python 3.8.13 with networkx==2.8.4 and numpy==1.22.4.


# How to Compile

```
sh compile.sh
```

# How to Run node2ket

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
./node2ket -net ./data/ca-GrQc-net.txt -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 1 -rho 0.1
```

node2ket+:
```
./node2ket -net ./data/ca-GrQc-net.txt -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -riemann 0 -print 1 -eval-nr 1 -rho 0.1 -config ca-GrQc-net.txt.louvain_config -thread 8
```


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