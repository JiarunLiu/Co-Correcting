echo `date`
echo -e "\e[1;46mTraining co-pencil new network\e[0m"
python Co-Correcting.py --dir experiment/paper/mnist/CoCor/clean --dataset 'mnist' --noise_type clean --noise 0.0 --optim "SGD"\
 --mix-grad 1 --discard 0  --forget-rate 0.01 --lambda1 500 --weight-decay 5e-4 --stage1 30 --stage2 140 --epochs 320
echo `date`
python Co-Correcting.py --dir experiment/paper/mnist/CoCor/0.05 --dataset 'mnist' --noise_type sn --noise 0.05 --optim "SGD"\
 --mix-grad 1 --discard 0  --forget-rate 0.05 --lambda1 1000 --weight-decay 5e-4 --stage1 30 --stage2 140 --epochs 320
echo `date`
python Co-Correcting.py --dir experiment/paper/mnist/CoCor/0.1 --dataset 'mnist' --noise_type sn --noise 0.1 --optim "SGD"\
 --mix-grad 1 --discard 0 --tips "experiment for paper"--forget-rate 0.1 --lambda1 2000 --weight-decay 5e-4 --stage1 30 --stage2 140 --epochs 320
echo `date`
python Co-Correcting.py --dir experiment/paper/mnist/CoCor/0.2 --dataset 'mnist' --noise_type sn --noise 0.2 --optim "SGD"\
 --mix-grad 1 --discard 0  --forget-rate 0.2 --lambda1 2500 --weight-decay 5e-4 --stage1 30 --stage2 140 --epochs 320
echo `date`
python Co-Correcting.py --dir experiment/paper/mnist/CoCor/0.3 --dataset 'mnist' --noise_type sn --noise 0.3 --optim "SGD"\
 --mix-grad 1 --discard 0  --forget-rate 0.3 --lambda1 3000 --weight-decay 5e-4 --stage1 30 --stage2 140 --epochs 320
echo `date`
python Co-Correcting.py --dir experiment/paper/mnist/CoCor/0.4 --dataset 'mnist' --noise_type sn --noise 0.4 --optim "SGD"\
 --mix-grad 1 --discard 0  --forget-rate 0.4 --lambda1 4000 --weight-decay 5e-4 --stage1 30 --stage2 140 --epochs 320
echo `date`
python Co-Correcting.py --dir experiment/paper/mnist/CoCor/pair_0.2 --dataset 'mnist' --noise_type pairflip --noise 0.2 --optim "SGD"\
 --mix-grad 1 --discard 0  --forget-rate 0.2 --lambda1 3000 --weight-decay 5e-4 --stage1 30 --stage2 140 --epochs 320
echo `date`
python Co-Correcting.py --dir experiment/paper/mnist/CoCor/pair_0.45 --dataset 'mnist' --noise_type pairflip --noise 0.45 --optim "SGD"\
 --mix-grad 1 --discard 0  --forget-rate 0.45 --lambda1 4000 --weight-decay 5e-4 --stage1 30 --stage2 140 --epochs 320
echo -e "\e[1;46mTraining Finished!\e[0m"
echo `date`