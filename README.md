# Co-Correcting

Official implementation of TMI 2021 paper *Co-Correcting: Noise-tolerant Medical Image Classification via collaborative Label Probability Estimation* [[paper](https://ieeexplore.ieee.org/document/9461766)][[arxiv](https://arxiv.org/abs/2109.05159)]

## Requirements:

+ python3.6
+ numpy
+ torch-1.4.0
+ torchvision-0.5.0

## Usage

`Co-Correcting.py` is used for both training a model on dataset with noisy labels and validating it.

Here is an example:

```shell
python Co-Correcting.py --dir ./experiment/ --dataset 'mnist' --noise_type sn --noise 0.2 --forget-rate 0.2
```

or you can train Co-Correcting with `.sh`:

```shell
sh script/mnist.sh
```

