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

data=/data/xionghao/data/PPI/PPI-net.txt
label=/data/xionghao/data/PPI/PPI-label.txt
vocab=/data/xionghao/data/PPI/PPI-vocab.txt
portion=0.5

# ./node2ket -net $data -dim 4 -C 2 -rw 1 -obj logistic -samples 100 -print 0 -riemann 0
# NCn2k
# NR
# ./node2ket -net $data -dim 4 -C 2 -rw 1 -obj mt -samples 100 -print 0

# ./node2ket -net $data -dim 8 -C 2 -rw 1 -obj logistic -samples 100 -print 0 -riemann 0
# NCn2k
# NR
# ./node2ket -net $data -dim 8 -C 2 -rw 1 -obj mt -samples 100 -print 0

# ./node2ket -net $data -dim 8 -C 4 -rw 1 -obj logistic -samples 100 -print 0 -riemann 0
# NCn2k
# NR
# ./node2ket -net $data -dim 8 -C 4 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 8 -C 8 -rw 1 -obj mt -samples 100 -print 0
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -obj mt -samples 100 -print 0


data=/data/xionghao/data/link_pred_data/PPI-net.txt-masked
# ./node2ket -net $data -dim 4 -C 2 -rw 1 -obj mt -samples 100 -print 0 -eval-nr 0 
# LPn2k
# ./node2ket -net $data -dim 8 -C 2 -rw 1 -obj mt -samples 100 -print 0 -eval-nr 0 
# LPn2k
# ./node2ket -net $data -dim 8 -C 4 -rw 1 -obj mt -samples 100 -print 0 -eval-nr 0 
# LPn2k
# ./node2ket -net $data -dim 8 -C 8 -rw 1 -obj mt -samples 100 -print 0 -eval-nr 0 
# LPn2k
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -obj mt -samples 100 -print 0 -eval-nr 0 
# LPn2k
./node2ket -net $data -dim 16 -C 16 -rw 1 -obj mt -samples 100 -print 0 -eval-nr 0 -rw 1 
LPn2k
./node2ket -net $data -dim 32 -C 16 -rw 1 -obj mt -samples 100 -print 0 -eval-nr 0 -rw 1 
LPn2k