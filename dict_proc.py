import numpy as np

label_dict = np.load('./label_dict.npy', allow_pickle=True)


def arr2str(arr):
    global label_dict
    strings = []
    for e in arr:
        strings.append(label_dict[e])
    return strings


def int2str(num):
    global label_dict
    return label_dict[num]
