import json
import numpy as np
from os.path import join


def check_label_acc(A, B, onehotA=False, onehotB=False):
    """
    get correct label percent in all labels
    :param A: label A
    :param B: label B
    :param onehotA: bool, is label A in onehot?
    :param onehotB: bool, is label B in onehot?
    :return: matched percent in total labels
    """
    A = np.argmax(A, axis=1) if onehotA else A
    B = np.argmax(B, axis=1) if onehotB else B

    try:
        assert A.shape == B.shape
    except:
        redux = min(A.shape[0], B.shape[0])
        A = A[:redux]
        B = B[:redux]

    t = np.sum(A == B)
    accu = t / len(A)
    return accu

def check_label_noisy2true(new_label, clean_label, noise_or_not, onehotA=False, onehotB=False):
    new_label = np.argmax(new_label, axis=1) if onehotA else new_label
    clean_label = np.argmax(clean_label, axis=1) if onehotB else clean_label

    try:
        assert new_label.shape == clean_label.shape
    except:
        redux = min(new_label.shape[0], clean_label.shape[0])
        new_label = new_label[:redux]
        clean_label = clean_label[:redux]

    assert new_label.shape == noise_or_not.shape
    assert new_label.shape == clean_label.shape

    n2t_num = np.sum((new_label == clean_label).astype(np.int32) * (~noise_or_not).astype(np.int32))
    n2t = n2t_num / clean_label.shape[0]

    return n2t


def check_label_true2noise(new_label, clean_label, noise_or_not, onehotA=False, onehotB=False):
    new_label = np.argmax(new_label, axis=1) if onehotA else new_label
    clean_label = np.argmax(clean_label, axis=1) if onehotB else clean_label

    try:
        assert new_label.shape == clean_label.shape
    except:
        redux = min(new_label.shape[0], clean_label.shape[0])
        new_label = new_label[:redux]
        clean_label = clean_label[:redux]

    assert new_label.shape == noise_or_not.shape
    assert new_label.shape == clean_label.shape

    t2n_num = np.sum((new_label != clean_label).astype(np.int32) * noise_or_not.astype(np.int32))
    t2n = t2n_num / clean_label.shape[0]

    return t2n

def check_label(new_label, clean_label, noise_or_not, onehotA=False, onehotB=False):
    acc = check_label_acc(new_label, clean_label, onehotA, onehotB)
    n2t = check_label_noisy2true(new_label, clean_label, noise_or_not, onehotA, onehotB)
    t2n = check_label_true2noise(new_label, clean_label, noise_or_not, onehotA, onehotB)
    return acc, n2t, t2n