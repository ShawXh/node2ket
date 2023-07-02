# Usage


## Generate random walk sequences

To generate node sequences for the file /data/ca-GrQc-net.txt by random walk:

```
python generate_random_walk.py --net /data/ca-GrQc-net.txt --n_walks 1 --walk_length 80
```

A local file /data/ca-GrQc-net.txt.rw_nwalks_1_length_80 will be generated.


## Prepare data for link prediction

To prepare the data for a complete network /data/ca-GrQc-net.txt, run the following command:
```
python link_pred_process.py --net /data/ca-GrQc-net.txt --ratio 0.01
```

Three files will be generated:

A file that contains the edges after removing a part of edges.
- /data/ca-GrQc-net.txt-masked

A file that contains the removed edges:
- /data/ca-GrQc-net.txt-masked-edges

A file that contains some noise edges by randomly matching the nodes, whose number is the same as the removed edges:
- /data/ca-GrQc-net.txt-noise-edges