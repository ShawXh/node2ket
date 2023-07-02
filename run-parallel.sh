# NR
data=/data/xionghao/data/ca-GrQc-net.txt
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -eval-nr 1 -thread 1
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -eval-nr 1 -thread 2
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -eval-nr 1 -thread 4
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -eval-nr 1 -thread 8
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -eval-nr 1 -thread 16

data=/data/xionghao/data/link_pred_data/ca-GrQc-net.txt-masked
./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2 -thread 1
python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2 -thread 2
python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2 -thread 4
python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2 -thread 8
python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt
./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 100 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2 -thread 16
python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt

# data=/data/xionghao/data/link_pred_data/PPI-net.txt-masked
# ./node2ket -net $data -dim 16 -C 8 -rw 1 -window-size 1 -obj mt -samples 10 -print 0 -eval-nr 0 -num-neg 5 -rho 0.2
# python ../eval_link_pred.py --net $data -t --tu-emb ./sub_embedding.txt