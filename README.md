# node2ket

This repo contain the code and data for the submission of node2ket to VLDB2024. The files include:


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

**Step 2 (optional for node2ket+).** Build index table for sub-embeddings. An example:
```
python ../InitTUConfigLouvain.py --net ./data/ca-GrQc-net.txt -L 1 -C 8 --res  100 500 1000 1500
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