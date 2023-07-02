function LPn2k() {
    cd ..
    python eval_link_pred.py --net ${data} --tu-emb ./build/sub_embedding.txt -t
    cd build
}

function NCn2k() {
    cd ../../Evaluate-Embedding-master
    ./to_binary -input ../node2ket/build/node_embedding.txt -output node_embedding.bin
    python2 test.py --emb ./node_embedding.bin --vocab ${vocab} --label ${label} --portion ${portion}
    cd ../node2ket/build
}

# link pred
# ./node2ket -net /data/xionghao/data/link_pred_data/ca-GrQc-net.txt-masked -dim 16 -C 8 -rw 1 -obj logistic -num-neg 5 -samples 50 -print 1
# cd ..
# python eval_link_pred.py --net /data/xionghao/data/link_pred_data/ca-GrQc-net.txt-masked --tu-emb ./build/sub_embedding.txt -t
# cd build

# ./node2ket -net /data/xionghao/data/link_pred_data/blog-net.txt-masked -dim 16 -C 8 -rw 1 -obj mt -samples 100 -print 1
# cd ..
# python eval_link_pred.py --net /data/xionghao/data/link_pred_data/blog-net.txt-masked --tu-emb ./build/sub_embedding.txt -t
# cd build

# ablation 1
function ablationLP() {
    ./node2ket -net $data -dim 4 -C 32 -rw 1 -obj mt -samples 100 -print 0
    LPn2k
    ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -print 0
    LPn2k
    ./node2ket -net $data -dim 16 -C 8 -rw 1 -obj mt -samples 100 -print 0
    LPn2k
    ./node2ket -net $data -dim 32 -C 4 -rw 1 -obj mt -samples 100 -print 0
    LPn2k
    ./node2ket -net $data -dim 64 -C 2 -rw 1 -obj mt -samples 100 -print 0
    LPn2k
    ./node2ket -net $data -dim 128 -C 1 -rw 1 -obj mt -samples 100 -print 0
    LPn2k
}
function ablationNC() {
    ./node2ket -net $data -dim 4 -C 32 -rw 1 -obj mt -samples 100 -print 0
    NCn2k
    ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -print 0
    NCn2k
    ./node2ket -net $data -dim 16 -C 8 -rw 1 -obj mt -samples 100 -print 0
    NCn2k
    ./node2ket -net $data -dim 32 -C 4 -rw 1 -obj mt -samples 100 -print 0
    NCn2k
    ./node2ket -net $data -dim 64 -C 2 -rw 1 -obj mt -samples 100 -print 0
    NCn2k
    ./node2ket -net $data -dim 128 -C 1 -rw 1 -obj mt -samples 100 -print 0
    NCn2k
}
# data=/data/xionghao/data/link_pred_data/youtube-sparse-net.txt-masked
# ./node2ket -net $data -dim 4 -C 32 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 32 -C 4 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 64 -C 2 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 128 -C 1 -rw 1 -obj mt -samples 100 -print 0
# LPn2k

# data=/data/xionghao/data/link_pred_data/PPI-net.txt-masked
# ablation1

# data=/data/xionghao/data/link_pred_data/youtube-sparse-net.txt-masked
# ./node2ket -net $data -dim 4 -C 16 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 8 -C 8 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 4 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 32 -C 2 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 64 -C 1 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 32 -C 1 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 8 -C 4 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 4 -C 8 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 1 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 8 -C 2 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 4 -C 4 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 8 -C 1 -rw 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 4 -C 2 -rw 1 -obj mt -samples 100 -print 0
# LPn2k



data=/data/xionghao/data/youtube/youtube-sparse-net.txt
vocab=/data/xionghao/data/youtube/youtube-sparse-vocab.txt
label=/data/xionghao/data/youtube/youtube-sparse-label.txt
# ./node2ket -net $data -dim 4 -C 4 -rw 1 -obj logistic -samples 100 -print 0
# portion=0.09
# NCn2k
# ./node2ket -net $data -dim 8 -C 2 -rw 1 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 1 -rw 1 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 8 -C 4 -rw 1 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 32 -C 1 -rw 1 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 32 -C 2 -rw 1 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 64 -C 1 -rw 1 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 64 -C 2 -rw 1 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 128 -C 1 -rw 1 -obj logistic -samples 100 -print 0
# NCn2k


# data=/data/xionghao/data/youtube/youtube-sparse-net.txt
# vocab=/data/xionghao/data/youtube/youtube-sparse-vocab.txt
# label=/data/xionghao/data/youtube/youtube-sparse-label.txt
# ./node2ket -net $data -dim 4 -C 4 -rw 1 -obj logistic -samples 100 -print 0
# portion=0.09
# NCn2k
# ./node2ket -net $data -dim 8 -C 2 -rw 1 -obj logistic -samples 100 -print 0
# portion=0.09
# NCn2k
# ./node2ket -net $data -dim 16 -C 1 -rw 1 -obj logistic -samples 100 -print 0
# portion=0.09
# NCn2k

# data=/data/xionghao/data/youtube/youtube-sparse-net.txt
# vocab=/data/xionghao/data/youtube/youtube-sparse-vocab.txt
# label=/data/xionghao/data/youtube/youtube-sparse-label.txt
# portion=0.09
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -window-size 2 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -window-size 3 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -window-size 4 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -window-size 5 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -window-size 6 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -window-size 7 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -window-size 8 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -window-size 9 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -window-size 10 -obj logistic -samples 100 -print 0
# NCn2k

# data=/data/xionghao/data/link_pred_data/youtube-sparse-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 2 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 3 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 4 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 5 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 6 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 7 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 8 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 9 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 10 -obj mt -samples 100 -print 0
# LPn2k

# data=/data/xionghao/data/youtube/youtube-sparse-net.txt
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 2 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 3 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 4 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 5 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 6 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 7 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 8 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 9 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 10 -obj mt -samples 100 -print 0

# data=/data/xionghao/data/youtube/youtube-sparse-net.txt
# vocab=/data/xionghao/data/youtube/youtube-sparse-vocab.txt
# label=/data/xionghao/data/youtube/youtube-sparse-label.txt
# portion=0.09
# ./node2ket -net $data -dim 16 -C 2 -rwr 1 -ppralpha 0.9 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rwr 1 -ppralpha 0.8 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rwr 1 -ppralpha 0.7 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rwr 1 -ppralpha 0.6 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rwr 1 -ppralpha 0.5 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rwr 1 -ppralpha 0.4 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rwr 1 -ppralpha 0.3 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rwr 1 -ppralpha 0.2 -obj logistic -samples 100 -print 0
# NCn2k
# ./node2ket -net $data -dim 16 -C 2 -rwr 1 -ppralpha 0.1 -obj logistic -samples 100 -print 0
# NCn2k

# data=/data/xionghao/data/link_pred_data/youtube-sparse-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.9 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.8 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.7 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.6 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.5 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.4 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.3 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.2 -obj mt -samples 100 -print 0
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.1 -obj mt -samples 100 -print 0
# LPn2k

# data=/data/xionghao/data/youtube/youtube-sparse-net.txt
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.9 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.8 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.7 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.6 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.5 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.4 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.3 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.2 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rwr 1 -ppralpha 0.1 -obj mt -samples 100 -print 0

# data=/data/xionghao/data/youtube/youtube-sparse-net.txt
# ./node2ket -net $data -dim 4 -C 32 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 32 -C 4 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 64 -C 2 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 128 -C 1 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 4 -C 16 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 8 -C 8 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 4 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 32 -C 2 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 64 -C 1 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 4 -C 8 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 8 -C 4 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 2 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 32 -C 1 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 4 -C 4 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 8 -C 2 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 1 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 4 -C 2 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 8 -C 1 -rw 1 -obj mt -samples 100 -print 0


# ablation study for constraints

# seed=0

# function ablation_constraint() {
#     data=/data/xionghao/data/link_pred_data/PPI-net.txt-masked
#     ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -print 0 -riemann 0 -seed $seed
#     LPn2k
#     ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -print 0 -riemann 1 -seed $seed
#     LPn2k
#     ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -print 0 -riemann 2 -seed $seed
#     LPn2k
#     ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -print 0 -riemann 0 -zero 0 -seed $seed
#     LPn2k

#     data=/data/xionghao/data/PPI/PPI-net.txt
#     vocab=/data/xionghao/data/PPI/PPI-vocab.txt
#     label=/data/xionghao/data/PPI/PPI-label.txt
#     portion=0.5
#     ./node2ket -net $data -dim 8 -C 2 -rw 1 -obj logistic -samples 100 -print 0 -riemann 0 -seed $seed
#     NCn2k
#     ./node2ket -net $data -dim 8 -C 2 -rw 1 -obj logistic -samples 100 -print 0 -riemann 1 -seed $seed
#     NCn2k
#     ./node2ket -net $data -dim 8 -C 2 -rw 1 -obj logistic -samples 100 -print 0 -riemann 2 -seed $seed
#     NCn2k
#     ./node2ket -net $data -dim 8 -C 2 -rw 1 -obj logistic -samples 100 -print 0 -riemann 0 -zero 0 -seed $seed
#     NCn2k

#     ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -print 0 -riemann 0 -seed $seed
#     ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -print 0 -riemann 1 -seed $seed
#     ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -print 0 -riemann 2 -seed $seed
#     ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -print 0 -riemann 0 -zero 0 -seed $seed
# }

# seed=1
# ablation_constraint
# seed=2
# ablation_constraint
# seed=3
# ablation_constraint
# seed=4
# ablation_constraint



# ablations study for optimizers
# convergence
data=/data/xionghao/data/youtube/youtube-sparse-net.txt
# ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 600 -opt sgd -rho 0.01 -log 1 -thread 1 -riemann 0
# mv ./tmpfile/log.txt ./sgd_log.txt
# ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 600 -opt rmsprop -log 1 -thread 1 -riemann 0
# mv ./tmpfile/log.txt ./rmsprop_log.txt
# ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 600 -opt adagrad -log 1 -thread 1 -riemann 0
# mv ./tmpfile/log.txt ./adagrad_log.txt

# ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj logistic -samples 500 -opt sgd -rho 0.01 -log 1 -thread 1 -riemann 0
# mv ./tmpfile/log.txt ./sgd_log_logistic.txt
# ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj logistic -samples 500 -opt rmsprop -log 1 -thread 1 -riemann 0
# mv ./tmpfile/log.txt ./rmsprop_log_logistic.txt
# ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj logistic -samples 500 -opt adagrad -log 1 -thread 1 -riemann 0
# mv ./tmpfile/log.txt ./adagrad_log_logistic.txt

# performance
# data=/data/xionghao/data/youtube/youtube-sparse-net.txt
# ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -opt sgd -rho 0.02 -riemann 2
# ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj mt -samples 100 -opt rmsprop  -riemann 2

# data=/data/xionghao/data/link_pred_data/youtube-sparse-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -rho 0.02 -opt sgd -window-size 2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -rho 0.2 -opt rmsprop -window-size 2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt

data=/data/xionghao/data/youtube/youtube-sparse-net.txt
vocab=/data/xionghao/data/youtube/youtube-sparse-vocab.txt
label=/data/xionghao/data/youtube/youtube-sparse-label.txt
# ./node2ket -net $data -dim 8 -C 4 -rw 1 -window-size 1 -obj logistic -samples 100 -opt sgd -print 1 -eval-nr 0 -riemann 0 -num-neg 5 -rho 0.02
# ./node2ket -net $data -dim 8 -C 4 -rw 1 -window-size 1 -obj logistic -samples 100 -opt adagrad -print 1 -eval-nr 0 -riemann 0 -num-neg 5 -rho 0.5
# portion=0.05
# NCn2k
# ./node2ket -net $data -dim 8 -C 4 -rw 1 -window-size 1 -obj logistic -samples 100 -opt rmsprop -print 1 -eval-nr 0 -riemann 0 -num-neg 5 -rho 0.2
# portion=0.05
# NCn2k



# ablation study for objectives
# NR
# data=/data/xionghao/data/youtube/youtube-sparse-net.txt
# ./node2ket -net $data -dim 8 -C 16 -rw 1 -obj logistic -samples 100 -rho 0.2

# LP
# data=/data/xionghao/data/link_pred_data/youtube-sparse-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 2 -obj logistic -samples 100 -num-neg 5 -rho 0.2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt

# NC
# ./node2ket -net $data -dim 8 -C 4 -rw 1 -window-size 1 -obj mt -samples 100 -print 1 -eval-nr 0 -riemann 0 -rho 0.5
# portion=0.05
# NCn2k
