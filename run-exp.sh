function NCn2k() {
    cd ../../Evaluate-Embedding-master
    ./to_binary -input ../node2ket/build/node_embedding.txt -output node_embedding.bin
    python2 test.py --emb ./node_embedding.bin --vocab ${vocab} --label ${label} --portion ${portion}
    cd ../node2ket/build
}

data=/data/xionghao/data/youtube/youtube-sparse-net.txt
vocab=/data/xionghao/data/youtube/youtube-sparse-vocab.txt
label=/data/xionghao/data/youtube/youtube-sparse-label.txt
# ./node2ket -net $data -dim 8 -C 2 -rw 1 -window-size 1 -obj logistic -samples 100 -print 1 -eval-nr 0 -riemann 0 -num-neg 5 -rho 0.2 -thread 8
# ./node2ket -net $data -dim 4 -C 4 -rw 1 -window-size 1 -obj logistic -samples 100 -print 1 -eval-nr 0 -riemann 0 -num-neg 5 -rho 0.2
# ./node2ket -net $data -dim 8 -C 4 -rw 1 -window-size 1 -obj logistic -samples 100 -print 1 -eval-nr 0 -riemann 0 -num-neg 5 -rho 0.2
# python ../InitTUConfigLouvain.py --net $data -L 1 -C 4 --res 1000 2000
# ./node2ket -net $data -dim 8 -C 4 -rw 1 -window-size 1 -obj logistic -samples 100 -print 1 -eval-nr 0 -riemann 0 -num-neg 5 -rho 0.2 -config youtube-sparse-net.txt.louvain_config -thread 8
# portion=0.01
# NCn2k
# portion=0.03
# NCn2k
# portion=0.05
# NCn2k
# portion=0.07
# NCn2k
# portion=0.09
# NCn2k

data=/data/xionghao/data/blog/blog-net.txt
vocab=/data/xionghao/data/blog/blog-vocab.txt
# label=/data/xionghao/data/blog/blog-label.txt
# ./node2ket -net $data -dim 8 -C 4 -rw 1 -window-size 1 -obj logistic -samples 100 -print 1 -eval-nr 0 -riemann 0 -num-neg 5 -rho 0.2 -opt sgd
# portion=0.1
# NCn2k
# portion=0.3
# NCn2k
# portion=0.5
# NCn2k
# portion=0.7
# NCn2k
# portion=0.9
# NCn2k

data=/data/xionghao/data/PPI/PPI-net.txt
vocab=/data/xionghao/data/PPI/PPI-vocab.txt
label=/data/xionghao/data/PPI/PPI-label.txt
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -window-size 1 -obj logistic -samples 100 -print 1 -eval-nr 0 -riemann 0
# portion=0.1
# NCn2k
# portion=0.3
# NCn2k
# portion=0.5
# NCn2k
# portion=0.7
# NCn2k
# portion=0.9
# NCn2k

# python ../InitTUConfigLouvain.py --net /data/xionghao/data/PPI/PPI-net.txt -L 1 -C 2 --res 1000
# ./node2ket -net /data/xionghao/data/PPI/PPI-net.txt -dim 16 -C 2 -rw 1 -obj logistic -samples 100 -riemann 0 -print 1 -eval-nr 0 -rho 0.1 -config PPI-net.txt.louvain_config -thread 8
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/PPI/PPI-net.txt -L 1 -C 4 --res 500 1000
# ./node2ket -net /data/xionghao/data/PPI/PPI-net.txt -dim 8 -C 4 -rw 1 -obj logistic -samples 100 -riemann 0 -print 1 -eval-nr 0 -rho 0.1 -config PPI-net.txt.louvain_config -thread 8
# ./node2ket -net /data/xionghao/data/PPI/PPI-net.txt -dim 8 -C 4 -rw 1 -obj logistic -samples 100 -riemann 1 -print 1 -eval-nr 0 -rho 0.1
# portion=0.1
# NCn2k
# portion=0.3
# NCn2k
# portion=0.5
# NCn2k
# portion=0.7
# NCn2k
# portion=0.9
# NCn2k

# NR
data=/data/xionghao/data/youtube/youtube-sparse-net.txt
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 1 -num-neg 5 -rho 0.1
# data=/data/xionghao/data/PPI/PPI-net.txt
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 1 -num-neg 5 -rho 0.1
# data=/data/xionghao/data/blog/blog-net.txt
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 1 -num-neg 5 -rho 0.1
# data=/data/xionghao/data/ca-GrQc-net.txt
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 1 -num-neg 5 -rho 0.1
# data=/data/xionghao/data/dblp-net.txt
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 1 -num-neg 5 -rho 0.1

# python ../InitTUConfigLouvain.py --net /data/xionghao/data/PPI/PPI-net.txt -L 1 -C 8 --res 100 500 1000 1500
# ./node2ket -net /data/xionghao/data/PPI/PPI-net.txt -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -riemann 0 -print 1 -eval-nr 1 -rho 0.1 -config PPI-net.txt.louvain_config -thread 8
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/ca-GrQc-net.txt -L 1 -C 8 --res  100 500 1000 1500
# ./node2ket -net /data/xionghao/data/ca-GrQc-net.txt -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -riemann 0 -print 1 -eval-nr 1 -rho 0.1 -config ca-GrQc-net.txt.louvain_config -thread 8
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/dblp-net.txt -L 1 -C 8 --res 100 500 1000 1500
# ./node2ket -net /data/xionghao/data/dblp-net.txt -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -riemann 0 -print 1 -eval-nr 1 -rho 0.1 -config dblp-net.txt.louvain_config -thread 8
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/blog/blog-net.txt -L 1 -C 8 --res  100 500 1000 1500
# ./node2ket -net /data/xionghao/data/blog/blog-net.txt -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -riemann 0 -print 1 -eval-nr 1 -rho 0.1 -config blog-net.txt.louvain_config -thread 8
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/youtube/youtube-sparse-net.txt -L 1 -C 8 --res 100 500 1000 1500
# ./node2ket -net /data/xionghao/data/youtube/youtube-sparse-net.txt -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -riemann 0 -print 1 -eval-nr 1 -rho 0.1 -config youtube-sparse-net.txt.louvain_config -thread 8

# LP
# data=/data/xionghao/data/link_pred_data/youtube-sparse-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
# data=/data/xionghao/data/link_pred_data/PPI-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
# data=/data/xionghao/data/link_pred_data/ca-GrQc-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
# data=/data/xionghao/data/link_pred_data/dblp-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
# data=/data/xionghao/data/link_pred_data/blog-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt

# data=/data/xionghao/data/link_pred_data/youtube-sparse-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 2 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
# data=/data/xionghao/data/link_pred_data/PPI-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 2 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
# data=/data/xionghao/data/link_pred_data/ca-GrQc-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 2 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
# data=/data/xionghao/data/link_pred_data/dblp-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 2 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
# data=/data/xionghao/data/link_pred_data/blog-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 2 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neßß 5 -rho 0.2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt

window_size=2
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/link_pred_data/PPI-net.txt-masked -L 1 -C 8 --res 100 500 1000 1500
# ./node2ket -net /data/xionghao/data/link_pred_data/PPI-net.txt-masked -dim 16 -C 8 -rw 1 -window-size $window_size -obj mt -samples 100 -riemann 0 -print 1 -eval-nr 0 -rho 0.1 -config PPI-net.txt-masked.louvain_config -thread 8
# python ../eval_link_pred.py --net /data/xionghao/data/link_pred_data/PPI-net.txt-masked -t --tu-emb ./sub_embedding.txt --tu-config PPI-net.txt-masked.louvain_config
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/link_pred_data/ca-GrQc-net.txt-masked -L 1 -C 8 --res  100 500 1000 1500
# ./node2ket -net /data/xionghao/data/link_pred_data/ca-GrQc-net.txt-masked -dim 16 -C 8 -rw 1 -window-size $window_size -obj mt -samples 100 -riemann 0 -print 1 -eval-nr 0 -rho 0.1 -config ca-GrQc-net.txt-masked.louvain_config -thread 8
# python ../eval_link_pred.py --net /data/xionghao/data/link_pred_data/ca-GrQc-net.txt-masked -t --tu-emb ./sub_embedding.txt --tu-config ca-GrQc-net.txt-masked.louvain_config
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/link_pred_data/dblp-net.txt-masked -L 1 -C 8 --res 100 500 1000 1500
# ./node2ket -net /data/xionghao/data/link_pred_data/dblp-net.txt-masked -dim 16 -C 8 -rw 1 -window-size $window_size -obj mt -samples 100 -riemann 0 -print 1 -eval-nr 0 -rho 0.1 -config dblp-net.txt-masked.louvain_config -thread 8
# python ../eval_link_pred.py --net /data/xionghao/data/link_pred_data/dblp-net.txt-masked -t --tu-emb ./sub_embedding.txt --tu-config dblp-net.txt-masked.louvain_config
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/link_pred_data/blog-net.txt-masked -L 1 -C 8 --res  100 500 1000 1500
# ./node2ket -net /data/xionghao/data/link_pred_data/blog-net.txt-masked -dim 16 -C 8 -rw 1 -window-size $window_size -obj mt -samples 100 -riemann 0 -print 1 -eval-nr 0 -rho 0.1 -config blog-net.txt-masked.louvain_config -thread 8
# python ../eval_link_pred.py --net /data/xionghao/data/link_pred_data/blog-net.txt-masked -t --tu-emb ./sub_embedding.txt --tu-config blog-net.txt-masked.louvain_config
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/link_pred_data/youtube-sparse-net.txt-masked -L 1 -C 8 --res 100 500 1000 1500
# ./node2ket -net /data/xionghao/data/link_pred_data/youtube-sparse-net.txt-masked -dim 16 -C 8 -rw 1 -window-size $window_size -obj mt -samples 100 -riemann 0 -print 0 -eval-nr 1 -rho 0.1 -config youtube-sparse-net.txt-masked.louvain_config -thread 8
# python ../eval_link_pred.py --net /data/xionghao/data/link_pred_data/youtube-sparse-net.txt-masked -t --tu-emb ./sub_embedding.txt --tu-config youtube-sparse-net.txt-masked.louvain_config



# Youtube

# NR
# node2ket
# ./node2ket -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -dim 8 -C 3 -obj mt -samples 1000 -rw 1 
# ./EvalNR -emb ./sub_embedding.txt -tensorized 1 -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt
# ./node2ket -seq /data/xionghao/data/youtube/youtube-idxnorm-net.txt-edgelist.rw -dim 8 -C 3 -obj mt -samples 1000 -rw 1 
# ./EvalNR -emb ./sub_embedding.txt -tensorized 1 -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt
# node2ket+
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -L 1 -C 3 --res 500 1000
# ./node2ket -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -dim 8 -C 3 -obj mt -samples 1000 -rw 1 -config youtube-idxnorm-net.txt.louvain_config
# ./EvalNR -emb ./sub_embedding.txt -tensorized 1 -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -config youtube-idxnorm-net.txt.louvain_config
# ./node2ket -seq /data/xionghao/data/youtube/youtube-idxnorm-net.txt-edgelist.rw -dim 8 -C 3 -obj mt -samples 1000 -rw 1 -config youtube-idxnorm-net.txt.louvain_config
# ./EvalNR -emb ./sub_embedding.txt -tensorized 1 -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -config youtube-idxnorm-net.txt.louvain_config

# better performance on NR
# ./node2ket -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -dim 8 -C 3 -obj mt -samples 1000 -rw 1 -rho 0.4 -window-size 2 -eval-nr 0
# ./EvalNR -emb ./sub_embedding.txt -tensorized 1 -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt
# ./node2ket -seq /data/xionghao/data/youtube/youtube-idxnorm-net.txt-edgelist.rw -dim 8 -C 3 -obj mt -samples 1000 -rw 1  -rho 0.4 -window-size 2 -eval-nr 0
# ./EvalNR -emb ./sub_embedding.txt -tensorized 1 -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt
# node2ket+
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -L 1 -C 3 --res 500 1000
# ./node2ket -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -dim 8 -C 3 -obj mt -samples 1000 -rw 1 -config youtube-idxnorm-net.txt.louvain_config  -rho 0.4 -window-size 2 -eval-nr 0
# ./EvalNR -emb ./sub_embedding.txt -tensorized 1 -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -config youtube-idxnorm-net.txt.louvain_config
# ./node2ket -seq /data/xionghao/data/youtube/youtube-idxnorm-net.txt-edgelist.rw -dim 8 -C 3 -obj mt -samples 1000 -rw 1 -config youtube-idxnorm-net.txt.louvain_config  -rho 0.4 -window-size 2 -eval-nr 0
# ./EvalNR -emb ./sub_embedding.txt -tensorized 1 -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -config youtube-idxnorm-net.txt.louvain_config


# compare with gensim and lightne
# function evallibn2k () {
#     declare starttime=`date +%s%N`
#     ./node2ket -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -dim 8 -C 3 -obj mt -samples $samples -rw 1 -thread 24 -eval-nr 0 -rho $rho
#     declare endtime=`date +%s%N`
#     c=`expr $endtime - $starttime`
#     c=`expr $c / 1000000`
#     echo "$c ms"
#     ./EvalNR -emb ./sub_embedding.txt -tensorized 1 -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt
# }
# rho=0.1
samples=1000
# # evallibn2k
# # samples=800
# # evallibn2k
# rho=0.2
# evallibn2k
rho=0.4
# evallibn2k
./node2ket -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -dim 8 -C 3 -obj mt -samples $samples -rw 1 -thread 1 -eval-nr 0 -rho 0.2 -rw 1 -window-size 1
# ./EvalNR -emb ./sub_embedding.txt -tensorized 1 -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt



# LP
# ./node2ket -net /data/xionghao/data/link_pred_data/youtube-idxnorm-net.txt-masked -dim 8 -C 3 -obj mt -samples 1000 -rw 1 -window-size 2
# python ../eval_link_pred.py --net /data/xionghao/data/link_pred_data/youtube-idxnorm-net.txt-masked -t --tu-emb ./sub_embedding.txt
# ./node2ket -seq /data/xionghao/data/link_pred_data/youtube-idxnorm-net.txt-masked-edgelist.rw_nwalks_1 -dim 8 -C 3 -obj mt -samples 1000 -rw 1 -window-size 2
# python ../eval_link_pred.py --net /data/xionghao/data/link_pred_data/youtube-idxnorm-net.txt-masked -t --tu-emb ./sub_embedding.txt
# python ../InitTUConfigLouvain.py --net /data/xionghao/data/link_pred_data/youtube-idxnorm-net.txt-masked -L 1 -C 3 --res 500 1000
# ./node2ket -net /data/xionghao/data/link_pred_data/youtube-idxnorm-net.txt-masked -dim 8 -C 3 -obj mt -samples 1000 -rw 1 -window-size 2 -config youtube-idxnorm-net.txt-masked.louvain_config
# python ../eval_link_pred.py --net /data/xionghao/data/link_pred_data/youtube-idxnorm-net.txt-masked -t --tu-emb ./sub_embedding.txt --tu-config youtube-idxnorm-net.txt-masked.louvain_config
# ./node2ket -seq /data/xionghao/data/link_pred_data/youtube-idxnorm-net.txt-masked-edgelist.rw_nwalks_1 -dim 8 -C 3 -obj mt -samples 1000 -rw 1 -window-size 2 -config youtube-idxnorm-net.txt-masked.louvain_config
# python ../eval_link_pred.py --net /data/xionghao/data/link_pred_data/youtube-idxnorm-net.txt-masked -t --tu-emb ./sub_embedding.txt --tu-config youtube-idxnorm-net.txt-masked.louvain_config

# NC
function NCYoutube() {
    cd /data/xionghao/Evaluate-Embedding-master
    ./to_binary -input ../node2ket/build/node_embedding.txt -output node_embedding.bin
    python2 test.py --emb node_embedding.bin --vocab ../data/youtube/youtube-idxnorm-vocab.txt --label ../data/youtube/youtube-idxnorm-labels.txt
    cd ../node2ket/build
}
# ./node2ket -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -dim 8 -C 3 -obj logistic -samples 1000 -rw 1 -rho 0.5
# ./node2ket -net /data/xionghao/data/youtube/youtube-idxnorm-net.txt -dim 8 -C 3 -obj logistic -samples 1000 -rw 1 -rho 0.5 -config youtube-idxnorm-net.txt-masked.louvain_config
# NCYoutube

# ./node2ket -seq /data/xionghao/data/youtube/youtube-idxnorm-net.txt-edgelist.rw -dim 8 -C 3 -obj logistic -samples 1000 -rw 1 -rho 0.5
# NCYoutube

# ./node2ket -seq /data/xionghao/data/youtube/youtube-idxnorm-net.txt-edgelist.rw -dim 8 -C 3 -obj logistic -samples 1000 -rw 1 -rho 0.5 -config youtube-idxnorm-net.txt-masked.louvain_config
# NCYoutube