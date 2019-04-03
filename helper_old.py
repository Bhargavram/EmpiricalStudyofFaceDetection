import gzip
import json
import numpy as np
from integral_image import (
    IntegralImage, OriginalFeatures,
)


def load_training_data(shuffle=True,size='19',dataset='colorferet'):
    X, Y = [], []
    if dataset == 'colorferet':
        if size == '19':
            pos, neg = 'data/colorferet/scaled_data_19x19.json.gz', \
                       'data/non-face/non-face_data_19x19.json.gz'
        else:
            pos, neg = 'data/colorferet/scaled_data_24x24.json.gz', \
                       'data/non-face/non-face_data_24x24.json.gz'
    elif dataset == 'mit+cmu':
        pos, neg = 'data/mit+cmu/train/train_face.json.gz', \
                   'data/mit+cmu/train/train_non-face.json.gz'
    else:
        raise ValueError

    with gzip.open(pos, 'rb') as fp:
        objs = json.load(fp)
        X += [np.array(image).reshape(19, 19) for image in objs.values()]
        Y += [1 for _ in range(len(objs))]

    with gzip.open(neg, 'rb') as fp:
        objs = json.load(fp)
        X += [np.array(image).reshape(19, 19) for image in objs.values()]
        Y += [0 for _ in range(len(objs))]

    if shuffle:
        s = [(x, y) for x, y in zip(X, Y)]
        np.random.shuffle(s)
        X, Y = [], []
        for (x, y) in s:
            X.append(x)
            Y.append(y)

    return np.asarray(X), np.asarray(Y)


def load_test_data(shuffle=True):
    X, Y = [], []

    with gzip.open('data/mit+cmu/test/test_face.json.gz', 'rb') as fp:
        objs = json.load(fp)
        X += [np.array(image).reshape(19, 19) for image in objs.values()]
        Y += [1 for _ in range(len(objs))]

    with gzip.open('data/mit+cmu/test/test_non-face.json.gz', 'rb') as fp:
        objs = json.load(fp)
        X += [np.array(image).reshape(19, 19) for image in objs.values()]
        Y += [0 for _ in range(len(objs))]

    if shuffle:
        s = [(x, y) for x, y in zip(X, Y)]
        np.random.shuffle(s)
        X, Y = [], []
        for (x, y) in s:
            X.append(x)
            Y.append(y)

    return np.asarray(X), np.asarray(Y)


def load_features(type='HAAR'):
    features = []
    if type == 'HAAR':
        for feature_class in OriginalFeatures:
            features += IntegralImage.generate_features((19, 19), feature_class)
    else:
        raise ValueError('Unknown type:', type)
    return features
