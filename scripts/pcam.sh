echo `date`
echo -e "\e[1;46mTraining co-pencil new network\e[0m"
python Co-Correcting.py --dir experiment/paper/pcam/full/clean --dataset 'pcam' --noise_type clean --noise 0.0 --optim "SGD"\
 --mix-grad 1 --discard 0 --forget-rate 0.05 --lambda1 50 --lr 1e-3 --lr2 1e-4
echo `date`
python Co-Correcting.py --dir experiment/paper/pcam/full/0.05 --dataset 'pcam' --noise_type sn --noise 0.05 --optim "SGD"\
 --mix-grad 1 --discard 0 --forget-rate 0.05 --lambda1 75 --lr 1e-3 --lr2 1e-4
echo `date`
python Co-Correcting.py --dir experiment/paper/pcam/full/0.1 --dataset 'pcam' --noise_type sn --noise 0.1 --optim "SGD"\
 --mix-grad 1 --discard 0 --forget-rate 0.1 --lambda1 100 --lr 1e-3 --lr2 1e-4
echo `date`
python Co-Correcting.py --dir experiment/paper/pcam/full/0.2 --dataset 'pcam' --noise_type sn --noise 0.2 --optim "SGD"\
 --mix-grad 1 --discard 0 --forget-rate 0.2 --lambda1 150 --lr 1e-3 --lr2 1e-4
echo `date`
python Co-Correcting.py --dir experiment/paper/pcam/full/0.3 --dataset 'pcam' --noise_type sn --noise 0.3 --optim "SGD"\
 --mix-grad 1 --discard 0 --forget-rate 0.3 --lambda1 175 --lr 1e-3 --lr2 1e-4
echo `date`
python Co-Correcting.py --dir experiment/paper/pcam/full/0.4 --dataset 'pcam' --noise_type sn --noise 0.4 --optim "SGD"\
 --mix-grad 1 --discard 0 --forget-rate 0.4 --lambda1 200 --lr 1e-3 --lr2 1e-4
echo -e "\e[1;46mTraining Finished!\e[0m"
echo `date`