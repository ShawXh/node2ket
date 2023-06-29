net=/data/xionghao/data/youtube/youtube-sparse-net.txt
net=/data/xionghao/data/PPI/PPI-net.txt

# emb=/data/xionghao/node2ket/baselines/HHNE/code/emb.txt
# emb=/data/xionghao/node2ket/baselines/LouvainNE/emb.txt
# emb=/data/xionghao/node2ket/baselines/ProNE/emb2.txt
# emb=/data/xionghao/node2ket/baselines/NetSMF/example/emb.txt
# emb=/data/xionghao/node2ket/baselines/verse-emb.txt
# emb=/data/xionghao/node2ket/baselines/node2vec-emb.txt
emb=/data/xionghao/node2ket/baselines/line-emb.txt

# ./to_binary -input ../node2ket/baselines/ProNE/emb2.txt -output node_embedding.bin
# ./to_binary -input ../node2ket/baselines/line-emb.txt -output node_embedding.bin
# ./to_binary -input ../node2ket/baselines/line-1st-emb.txt -output node_embedding.bin
# ./to_binary -input ../node2ket/baselines/node2vec-emb.txt -output node_embedding.bin
# ./to_binary -input ../RNCE/tmpfile/emb_u -output node_embedding.bin
# ./to_binary -input ../node2ket/baselines/verse-emb.txt -output node_embedding.bin
# ./to_binary -input ../node2ket/baselines/LouvainNE/emb.txt -output node_embedding.bin
# ./to_binary -input ../node2ket/baselines/NetSMF/example/emb.txt -output node_embedding.bin
# ./to_binary -input ../node2ket/tmpfile/node_embedding.txt -output node_embedding.bin
# ./to_binary -input ../node2ket/baselines/HHNE/code/emb.txt -output node_embedding.bin

python network_reconstruction.py --net $net --emb1 $emb --emb2 $emb --func euc