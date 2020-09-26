echo `date`
echo -e "\e[1;46mTraining co-pencil new network\e[0m"
python Co-Correcting.py --dir experiment/paper/isic/CoCor/clean --dataset 'isic' --noise_type clean --noise 0.0 --optim "SGD"\
 --mix-grad 1 --discard 0 --forget-rate 0.05 --lambda1 50
echo `date`
python Co-Correcting.py --dir experiment/paper/isic/CoCor/0.05 --dataset 'isic' --noise_type sn --noise 0.05 --optim "SGD"\
 --mix-grad 1 --discard 0 --forget-rate 0.05 --lambda1 75
echo `date`
python Co-Correcting.py --dir experiment/paper/isic/CoCor/0.1 --dataset 'isic' --noise_type sn --noise 0.1 --optim "SGD"\
 --mix-grad 1 --discard 0 --forget-rate 0.1 --lambda1 100
echo `date`
python Co-Correcting.py --dir experiment/paper/isic/CoCor/0.2 --dataset 'isic' --noise_type sn --noise 0.2 --optim "SGD"\
 --mix-grad 1 --discard 0 --forget-rate 0.2 --lambda1 150
echo `date`
python Co-Correcting.py --dir experiment/paper/isic/CoCor/0.3 --dataset 'isic' --noise_type sn --noise 0.3 --optim "SGD"\
 --mix-grad 1 --discard 0 --forget-rate 0.3 --lambda1 175
echo `date`
python Co-Correcting.py --dir experiment/paper/isic/CoCor/0.4 --dataset 'isic' --noise_type sn --noise 0.4 --optim "SGD"\
 --mix-grad 1 --discard 0 --forget-rate 0.4 --lambda1 200
echo -e "\e[1;46mTraining Finished!\e[0m"
echo `date`