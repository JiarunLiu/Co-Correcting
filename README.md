# Co-Correcting

PyTorch implementation of *Co-Correcting:Noise-tolerant Medical Image Classification via collaborative Label Probability Estimation and Annotation Correction Curriculum*

## Requirements:

+ python3.6
+ numpy
+ torch-1.4.0
+ torchvision-0.5.0

## Usage

`Co-Correcting.py` is used for both training a model on dataset with noisy labels and validating it.

Here is an example:

```shell
python python Co-Correcting.py --dir experiment/ --dataset 'mnist' --noise_type sn --noise 0.2 --forget-rate 0.2
```

or you can train Co-Correcting with shel script:

```shell
sh script/mnist.sh
```

