import gzip
import json
import numpy as np
from integral_image import *
import matplotlib.pyplot as plt


def load_training_data(shuffle=True, dataset='colorferet'):
    X, Y, X_test, Y_test = [], [], [], []
    if dataset == 'colorferet':
        pos, neg = 'data/colorferet/scaled_data_19x19.json.gz', \
                   'data/non-face/non-face_data_19x19.json.gz'
    elif dataset == 'mit+cmu':
        pos, neg = 'data/mit+cmu/train/train_face.json.gz', \
                   'data/mit+cmu/train/train_non-face.json.gz'
    else:
        raise ValueError

    with gzip.open(pos, 'rb') as fp:
        objs = json.load(fp)
        X += [np.array(image).reshape(19, 19) for image in objs.values()]
        Y += [1 for _ in range(len(objs))]
    if dataset == 'colorferet':
        X_test = X[-500:]
        Y_test = Y[-500:]
        X = X[:-500]
        Y = Y[:-500]

    with gzip.open(neg, 'rb') as fp:
        objs = json.load(fp)
        X += [np.array(image).reshape(19, 19) for image in objs.values()]
        Y += [0 for _ in range(len(objs))]
    #plt.imshow(X[0],cmap="gray")
    #plt.show()
    if shuffle:
        s = [(x, y) for x, y in zip(X, Y)]
        np.random.shuffle(s)
        X, Y = [], []
        for (x, y) in s:
            X.append(x)
            Y.append(y)

    return np.asarray(X), np.asarray(Y), np.asarray(X_test), np.asarray(Y_test)


def load_test_data(shuffle=True,dataset='colorferet'):
    X, Y = [], []
    i = 0
    if dataset == 'colorferet':
        with gzip.open('data/mit+cmu/train/train_non-face.json.gz', 'rb') as fp:
            objs = json.load(fp)
            X += [np.array(image).reshape(19, 19) for image in objs.values()]
            Y += [0 for _ in range(len(objs))]
    else:
        with gzip.open('data/mit+cmu/test/test_face.json.gz', 'rb') as fp:
            objs = json.load(fp)
            X += [np.array(image).reshape(19, 19) for image in objs.values()]
            Y += [1 for _ in range(len(objs))]
        #data/mit+cmu/train/train_non-face.json.gz
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
        return features
    elif type == 'HAAR-pre_defined':
        return [TwoRectangleVerticalFeature(15, 3, 4, 8),
                TwoRectangleHorizontalFeature(15, 3, 4, 8),
                FourRectangleFeature(15, 3, 4, 8),
                TwoRectangleHorizontalFeature(14, 3, 5, 10),
                TwoRectangleVerticalFeature(15, 3, 4, 11),
                TwoRectangleVerticalFeature(15, 3, 4, 10),
                TwoRectangleHorizontalFeature(15, 3, 4, 10),
                FourRectangleFeature(15, 3, 4, 10),
                TwoRectangleVerticalFeature(14, 3, 4, 13),
                TwoRectangleVerticalFeature(14, 3, 4, 12),
                TwoRectangleHorizontalFeature(14, 3, 4, 12),
                ThreeRectangleHorizontalFeature(14, 3, 4, 12),
                FourRectangleFeature(14, 3, 4, 12),
                TwoRectangleVerticalFeature(14, 3, 4, 11),
                TwoRectangleVerticalFeature(15, 3, 4, 5),
                TwoRectangleVerticalFeature(15, 3, 4, 12),
                TwoRectangleHorizontalFeature(15, 3, 4, 12),
                ThreeRectangleHorizontalFeature(15, 3, 4, 12),
                FourRectangleFeature(15, 3, 4, 12),
                TwoRectangleVerticalFeature(15, 4, 4, 4),
                TwoRectangleHorizontalFeature(15, 4, 4, 4),
                FourRectangleFeature(15, 4, 4, 4),
                TwoRectangleVerticalFeature(14, 3, 4, 10),
                TwoRectangleHorizontalFeature(14, 3, 4, 10),
                FourRectangleFeature(14, 3, 4, 10),
                ThreeRectangleHorizontalFeature(14, 3, 5, 9),
                TwoRectangleVerticalFeature(15, 4, 4, 9),
                ThreeRectangleHorizontalFeature(15, 4, 4, 9),
                TwoRectangleVerticalFeature(6, 3, 12, 15),
                ThreeRectangleVerticalFeature(6, 3, 12, 15),
                ThreeRectangleHorizontalFeature(6, 3, 12, 15),
                TwoRectangleVerticalFeature(11, 3, 8, 12),
                TwoRectangleHorizontalFeature(11, 3, 8, 12),
                ThreeRectangleHorizontalFeature(11, 3, 8, 12),
                FourRectangleFeature(11, 3, 8, 12),
                ThreeRectangleHorizontalFeature(5, 3, 13, 15),
                TwoRectangleVerticalFeature(15, 3, 4, 9),
                ThreeRectangleHorizontalFeature(15, 3, 4, 9),
                ThreeRectangleVerticalFeature(4, 3, 15, 15),
                ThreeRectangleHorizontalFeature(4, 3, 15, 15),
                TwoRectangleVerticalFeature(15, 3, 4, 6),
                TwoRectangleHorizontalFeature(15, 3, 4, 6),
                ThreeRectangleHorizontalFeature(15, 3, 4, 6),
                FourRectangleFeature(15, 3, 4, 6),
                TwoRectangleVerticalFeature(15, 4, 4, 5),
                TwoRectangleVerticalFeature(4, 5, 4, 4),
                TwoRectangleHorizontalFeature(4, 5, 4, 4),
                FourRectangleFeature(4, 5, 4, 4),
                TwoRectangleHorizontalFeature(11, 3, 7, 14),
                TwoRectangleVerticalFeature(11, 4, 8, 5),
                TwoRectangleHorizontalFeature(12, 3, 7, 12),
                ThreeRectangleHorizontalFeature(12, 3, 7, 12),
                TwoRectangleHorizontalFeature(12, 5, 7, 4),
                TwoRectangleVerticalFeature(12, 3, 6, 11),
                ThreeRectangleVerticalFeature(12, 3, 6, 11),
                TwoRectangleVerticalFeature(5, 3, 14, 15),
                ThreeRectangleHorizontalFeature(5, 3, 14, 15),
                TwoRectangleVerticalFeature(3, 3, 14, 15),
                ThreeRectangleHorizontalFeature(3, 3, 14, 15),
                TwoRectangleVerticalFeature(4, 3, 14, 15),
                ThreeRectangleHorizontalFeature(4, 3, 14, 15),
                TwoRectangleVerticalFeature(14, 3, 4, 9),
                ThreeRectangleHorizontalFeature(14, 3, 4, 9),
                ThreeRectangleHorizontalFeature(4, 3, 13, 15),
                TwoRectangleVerticalFeature(11, 5, 4, 4),
                TwoRectangleHorizontalFeature(11, 5, 4, 4),
                FourRectangleFeature(11, 5, 4, 4),
                TwoRectangleVerticalFeature(10, 3, 8, 11),
                TwoRectangleVerticalFeature(13, 3, 6, 12),
                TwoRectangleHorizontalFeature(13, 3, 6, 12),
                ThreeRectangleVerticalFeature(13, 3, 6, 12),
                ThreeRectangleHorizontalFeature(13, 3, 6, 12),
                FourRectangleFeature(13, 3, 6, 12),
                TwoRectangleVerticalFeature(11, 3, 8, 10),
                TwoRectangleHorizontalFeature(11, 3, 8, 10),
                FourRectangleFeature(11, 3, 8, 10),
                TwoRectangleHorizontalFeature(14, 3, 5, 6),
                ThreeRectangleHorizontalFeature(14, 3, 5, 6),
                TwoRectangleHorizontalFeature(14, 3, 5, 8),
                TwoRectangleVerticalFeature(13, 3, 4, 15),
                ThreeRectangleHorizontalFeature(13, 3, 4, 15),
                TwoRectangleVerticalFeature(13, 5, 6, 4),
                TwoRectangleHorizontalFeature(13, 5, 6, 4),
                ThreeRectangleVerticalFeature(13, 5, 6, 4),
                FourRectangleFeature(13, 5, 6, 4),
                TwoRectangleHorizontalFeature(13, 3, 5, 10),
                TwoRectangleHorizontalFeature(7, 3, 11, 14),
                TwoRectangleVerticalFeature(13, 3, 6, 11),
                ThreeRectangleVerticalFeature(13, 3, 6, 11),
                ThreeRectangleHorizontalFeature(11, 3, 7, 15),
                TwoRectangleVerticalFeature(14, 3, 4, 6),
                TwoRectangleHorizontalFeature(14, 3, 4, 6),
                ThreeRectangleHorizontalFeature(14, 3, 4, 6),
                FourRectangleFeature(14, 3, 4, 6),
                TwoRectangleHorizontalFeature(14, 3, 5, 12),
                ThreeRectangleHorizontalFeature(14, 3, 5, 12),
                TwoRectangleHorizontalFeature(11, 5, 7, 4),
                TwoRectangleVerticalFeature(15, 3, 4, 7),
                TwoRectangleHorizontalFeature(6, 3, 13, 12),
                ThreeRectangleHorizontalFeature(6, 3, 13, 12),
                ThreeRectangleVerticalFeature(10, 3, 9, 11),
                TwoRectangleVerticalFeature(12, 3, 6, 12),
                TwoRectangleHorizontalFeature(12, 3, 6, 12),
                ThreeRectangleVerticalFeature(12, 3, 6, 12),
                ThreeRectangleHorizontalFeature(12, 3, 6, 12),
                FourRectangleFeature(12, 3, 6, 12),
                TwoRectangleVerticalFeature(10, 3, 8, 13),
                TwoRectangleVerticalFeature(13, 3, 6, 9),
                TwoRectangleVerticalFeature(13, 3, 6, 10),
                TwoRectangleHorizontalFeature(13, 3, 6, 10),
                ThreeRectangleVerticalFeature(13, 3, 6, 9),
                ThreeRectangleVerticalFeature(13, 3, 6, 10),
                ThreeRectangleHorizontalFeature(13, 3, 6, 9),
                FourRectangleFeature(13, 3, 6, 10),
                TwoRectangleHorizontalFeature(4, 5, 15, 4),
                ThreeRectangleVerticalFeature(4, 5, 15, 4),
                TwoRectangleVerticalFeature(12, 3, 6, 15),
                ThreeRectangleVerticalFeature(12, 3, 6, 15),
                ThreeRectangleHorizontalFeature(12, 3, 6, 15),
                TwoRectangleVerticalFeature(14, 3, 4, 8),
                TwoRectangleHorizontalFeature(14, 3, 4, 8),
                FourRectangleFeature(14, 3, 4, 8),
                TwoRectangleVerticalFeature(11, 3, 8, 11),
                TwoRectangleVerticalFeature(11, 5, 8, 4),
                TwoRectangleHorizontalFeature(11, 5, 8, 4),
                FourRectangleFeature(11, 5, 8, 4),
                TwoRectangleVerticalFeature(6, 12, 10, 7),
                TwoRectangleVerticalFeature(5, 3, 14, 12),
                TwoRectangleHorizontalFeature(5, 3, 14, 12),
                ThreeRectangleHorizontalFeature(5, 3, 14, 12),
                FourRectangleFeature(5, 3, 14, 12),
                TwoRectangleVerticalFeature(10, 3, 8, 15),
                ThreeRectangleHorizontalFeature(10, 3, 8, 15),
                TwoRectangleVerticalFeature(8, 3, 10, 14),
                TwoRectangleHorizontalFeature(8, 3, 10, 14),
                FourRectangleFeature(8, 3, 10, 14),
                TwoRectangleHorizontalFeature(10, 3, 9, 10),
                ThreeRectangleVerticalFeature(10, 3, 9, 10),
                TwoRectangleVerticalFeature(5, 3, 14, 13),
                TwoRectangleHorizontalFeature(7, 13, 5, 4),
                TwoRectangleHorizontalFeature(13, 3, 5, 12),
                ThreeRectangleHorizontalFeature(13, 3, 5, 12),
                TwoRectangleVerticalFeature(4, 5, 14, 4),
                TwoRectangleHorizontalFeature(4, 5, 14, 4),
                FourRectangleFeature(4, 5, 14, 4),
                TwoRectangleVerticalFeature(12, 4, 6, 4),
                TwoRectangleHorizontalFeature(12, 4, 6, 4),
                ThreeRectangleVerticalFeature(12, 4, 6, 4),
                FourRectangleFeature(12, 4, 6, 4),
                TwoRectangleVerticalFeature(13, 3, 6, 8),
                TwoRectangleHorizontalFeature(13, 3, 6, 8),
                ThreeRectangleVerticalFeature(13, 3, 6, 8),
                FourRectangleFeature(13, 3, 6, 8),
                TwoRectangleHorizontalFeature(12, 3, 7, 6),
                ThreeRectangleHorizontalFeature(12, 3, 7, 6),
                TwoRectangleHorizontalFeature(11, 3, 7, 10),
                TwoRectangleVerticalFeature(3, 12, 10, 7),
                TwoRectangleHorizontalFeature(4, 5, 5, 4),
                TwoRectangleHorizontalFeature(13, 5, 5, 4),
                ThreeRectangleVerticalFeature(5, 12, 9, 7),
                TwoRectangleVerticalFeature(15, 4, 4, 8),
                TwoRectangleHorizontalFeature(15, 4, 4, 8),
                FourRectangleFeature(15, 4, 4, 8),
                TwoRectangleHorizontalFeature(11, 3, 7, 12),
                ThreeRectangleHorizontalFeature(11, 3, 7, 12),
                TwoRectangleHorizontalFeature(3, 13, 13, 6),
                ThreeRectangleHorizontalFeature(3, 13, 13, 6),
                TwoRectangleHorizontalFeature(6, 14, 5, 4),
                TwoRectangleHorizontalFeature(6, 4, 7, 14),
                TwoRectangleHorizontalFeature(10, 5, 9, 4),
                ThreeRectangleVerticalFeature(10, 5, 9, 4),
                TwoRectangleVerticalFeature(4, 4, 14, 12),
                TwoRectangleHorizontalFeature(4, 4, 14, 12),
                ThreeRectangleHorizontalFeature(4, 4, 14, 12),
                FourRectangleFeature(4, 4, 14, 12),
                TwoRectangleVerticalFeature(7, 3, 12, 10),
                TwoRectangleHorizontalFeature(7, 3, 12, 10),
                ThreeRectangleVerticalFeature(7, 3, 12, 10),
                FourRectangleFeature(7, 3, 12, 10),
                ThreeRectangleHorizontalFeature(10, 3, 7, 15),
                TwoRectangleVerticalFeature(5, 5, 14, 4),
                TwoRectangleHorizontalFeature(5, 5, 14, 4),
                FourRectangleFeature(5, 5, 14, 4),
                TwoRectangleVerticalFeature(7, 12, 4, 6),
                TwoRectangleHorizontalFeature(7, 12, 4, 6),
                ThreeRectangleHorizontalFeature(7, 12, 4, 6),
                FourRectangleFeature(7, 12, 4, 6),
                TwoRectangleVerticalFeature(9, 3, 10, 11),
                TwoRectangleVerticalFeature(12, 5, 6, 4),
                TwoRectangleHorizontalFeature(12, 5, 6, 4),
                ThreeRectangleVerticalFeature(12, 5, 6, 4),
                ThreeRectangleVerticalFeature(3, 3, 15, 15),
                ThreeRectangleHorizontalFeature(3, 3, 15, 15),
                FourRectangleFeature(12, 5, 6, 4),
                TwoRectangleHorizontalFeature(5, 5, 13, 4),
                TwoRectangleHorizontalFeature(12, 3, 7, 10),
                TwoRectangleVerticalFeature(6, 12, 6, 6),
                TwoRectangleHorizontalFeature(6, 12, 6, 6),
                ThreeRectangleVerticalFeature(6, 12, 6, 6),
                ThreeRectangleHorizontalFeature(6, 12, 6, 6),
                FourRectangleFeature(6, 12, 6, 6),
                ThreeRectangleHorizontalFeature(12, 3, 7, 9),
                ThreeRectangleHorizontalFeature(6, 4, 7, 15),
                TwoRectangleVerticalFeature(4, 13, 4, 6),
                TwoRectangleHorizontalFeature(4, 13, 4, 6),
                ThreeRectangleHorizontalFeature(4, 13, 4, 6),
                FourRectangleFeature(4, 13, 4, 6),
                TwoRectangleVerticalFeature(6, 4, 12, 12),
                TwoRectangleHorizontalFeature(6, 4, 12, 12),
                ThreeRectangleVerticalFeature(6, 4, 12, 12),
                ThreeRectangleHorizontalFeature(6, 4, 12, 12),
                FourRectangleFeature(6, 4, 12, 12),
                ThreeRectangleHorizontalFeature(6, 3, 11, 15),
                TwoRectangleVerticalFeature(6, 13, 6, 4),
                TwoRectangleVerticalFeature(6, 5, 12, 4),
                TwoRectangleHorizontalFeature(6, 13, 6, 4),
                TwoRectangleHorizontalFeature(6, 5, 12, 4),
                ThreeRectangleVerticalFeature(6, 13, 6, 4),
                ThreeRectangleVerticalFeature(6, 5, 12, 4),
                FourRectangleFeature(6, 13, 6, 4),
                FourRectangleFeature(6, 5, 12, 4),
                TwoRectangleVerticalFeature(7, 12, 10, 7),
                TwoRectangleHorizontalFeature(14, 4, 5, 4),
                ThreeRectangleHorizontalFeature(6, 3, 13, 15),
                TwoRectangleVerticalFeature(6, 12, 8, 7),
                TwoRectangleVerticalFeature(8, 12, 4, 6),
                TwoRectangleHorizontalFeature(8, 12, 4, 6),
                TwoRectangleHorizontalFeature(3, 3, 15, 14),
                ThreeRectangleVerticalFeature(3, 3, 15, 14),
                ThreeRectangleHorizontalFeature(8, 12, 4, 6),
                FourRectangleFeature(8, 12, 4, 6),
                TwoRectangleVerticalFeature(14, 3, 4, 14),
                TwoRectangleHorizontalFeature(14, 3, 4, 14),
                FourRectangleFeature(14, 3, 4, 14),
                TwoRectangleHorizontalFeature(3, 14, 9, 4),
                ThreeRectangleVerticalFeature(3, 14, 9, 4),
                ThreeRectangleHorizontalFeature(7, 3, 11, 15),
                ThreeRectangleHorizontalFeature(3, 4, 13, 15),
                TwoRectangleVerticalFeature(15, 3, 4, 13),
                TwoRectangleVerticalFeature(3, 14, 10, 5),
                TwoRectangleHorizontalFeature(4, 3, 15, 14),
                ThreeRectangleVerticalFeature(4, 3, 15, 14),
                ThreeRectangleHorizontalFeature(13, 3, 5, 15),
                TwoRectangleVerticalFeature(8, 13, 8, 6),
                TwoRectangleHorizontalFeature(8, 13, 8, 6),
                ThreeRectangleHorizontalFeature(8, 13, 8, 6),
                FourRectangleFeature(8, 13, 8, 6),
                TwoRectangleHorizontalFeature(3, 4, 15, 12),
                ThreeRectangleVerticalFeature(3, 4, 15, 12),
                ThreeRectangleHorizontalFeature(3, 4, 15, 12),
                TwoRectangleVerticalFeature(6, 4, 12, 13),
                ThreeRectangleVerticalFeature(6, 4, 12, 13),
                TwoRectangleVerticalFeature(7, 13, 10, 5),
                TwoRectangleVerticalFeature(3, 4, 14, 13),
                TwoRectangleVerticalFeature(3, 15, 12, 4),
                TwoRectangleHorizontalFeature(3, 15, 12, 4),
                ThreeRectangleVerticalFeature(7, 12, 9, 7),
                ThreeRectangleVerticalFeature(3, 15, 12, 4),
                FourRectangleFeature(3, 15, 12, 4),
                TwoRectangleVerticalFeature(14, 3, 4, 7),
                TwoRectangleVerticalFeature(6, 3, 12, 14),
                TwoRectangleHorizontalFeature(6, 3, 12, 14),
                ThreeRectangleVerticalFeature(6, 3, 12, 14),
                FourRectangleFeature(6, 3, 12, 14),
                TwoRectangleHorizontalFeature(3, 5, 5, 4),
                TwoRectangleVerticalFeature(5, 4, 4, 15),
                ThreeRectangleVerticalFeature(7, 13, 9, 5),
                ThreeRectangleHorizontalFeature(5, 4, 4, 15),
                TwoRectangleHorizontalFeature(6, 5, 11, 14),
                TwoRectangleHorizontalFeature(13, 3, 5, 14),
                TwoRectangleVerticalFeature(8, 3, 10, 15),
                ThreeRectangleHorizontalFeature(8, 3, 10, 15),
                TwoRectangleVerticalFeature(7, 12, 6, 5),
                TwoRectangleHorizontalFeature(7, 12, 5, 6),
                ThreeRectangleVerticalFeature(7, 12, 6, 5),
                ThreeRectangleHorizontalFeature(7, 12, 5, 6),
                TwoRectangleVerticalFeature(3, 5, 12, 14),
                TwoRectangleHorizontalFeature(3, 5, 12, 14),
                ThreeRectangleVerticalFeature(3, 5, 12, 14),
                FourRectangleFeature(3, 5, 12, 14),
                TwoRectangleHorizontalFeature(4, 5, 11, 4),
                TwoRectangleVerticalFeature(15, 4, 4, 10),
                TwoRectangleHorizontalFeature(15, 4, 4, 10),
                FourRectangleFeature(15, 4, 4, 10),
                TwoRectangleHorizontalFeature(5, 5, 13, 14),
                TwoRectangleVerticalFeature(11, 3, 8, 6),
                TwoRectangleHorizontalFeature(11, 3, 8, 6),
                ThreeRectangleHorizontalFeature(11, 3, 8, 6),
                FourRectangleFeature(11, 3, 8, 6),
                TwoRectangleVerticalFeature(3, 13, 14, 6),
                TwoRectangleHorizontalFeature(3, 13, 14, 6),
                ThreeRectangleHorizontalFeature(3, 13, 14, 6),
                FourRectangleFeature(3, 13, 14, 6),
                TwoRectangleVerticalFeature(8, 4, 10, 12),
                TwoRectangleVerticalFeature(6, 3, 12, 13),
                TwoRectangleHorizontalFeature(8, 4, 10, 12),
                ThreeRectangleVerticalFeature(6, 3, 12, 13),
                ThreeRectangleHorizontalFeature(8, 4, 10, 12),
                FourRectangleFeature(8, 4, 10, 12),
                TwoRectangleHorizontalFeature(3, 15, 13, 4),
                TwoRectangleVerticalFeature(7, 13, 6, 4),
                TwoRectangleHorizontalFeature(7, 13, 6, 4),
                ThreeRectangleVerticalFeature(7, 13, 6, 4),
                FourRectangleFeature(7, 13, 6, 4),
                TwoRectangleHorizontalFeature(5, 3, 13, 14),
                ThreeRectangleHorizontalFeature(3, 4, 5, 15),
                TwoRectangleVerticalFeature(13, 3, 6, 7),
                ThreeRectangleVerticalFeature(13, 3, 6, 7),
                TwoRectangleVerticalFeature(10, 3, 8, 14),
                TwoRectangleHorizontalFeature(10, 3, 8, 14),
                FourRectangleFeature(10, 3, 8, 14),
                TwoRectangleVerticalFeature(7, 12, 8, 7),
                TwoRectangleHorizontalFeature(11, 4, 7, 14),
                TwoRectangleVerticalFeature(7, 3, 10, 15),
                ThreeRectangleHorizontalFeature(7, 3, 10, 15),
                TwoRectangleVerticalFeature(5, 4, 8, 15),
                ThreeRectangleHorizontalFeature(5, 4, 8, 15),
                TwoRectangleHorizontalFeature(3, 5, 13, 14),
                TwoRectangleHorizontalFeature(13, 3, 5, 8),
                TwoRectangleVerticalFeature(7, 3, 8, 14),
                TwoRectangleHorizontalFeature(7, 3, 8, 14),
                FourRectangleFeature(7, 3, 8, 14),
                TwoRectangleVerticalFeature(13, 3, 6, 6),
                TwoRectangleHorizontalFeature(13, 3, 6, 6),
                ThreeRectangleVerticalFeature(13, 3, 6, 6),
                ThreeRectangleHorizontalFeature(13, 3, 6, 6),
                FourRectangleFeature(13, 3, 6, 6),
                TwoRectangleVerticalFeature(5, 12, 10, 7),
                ThreeRectangleHorizontalFeature(14, 4, 5, 9),
                TwoRectangleVerticalFeature(5, 4, 14, 13),
                TwoRectangleVerticalFeature(5, 4, 14, 14),
                TwoRectangleHorizontalFeature(5, 4, 14, 14),
                ThreeRectangleHorizontalFeature(7, 4, 7, 15),
                FourRectangleFeature(5, 4, 14, 14),
                TwoRectangleVerticalFeature(12, 4, 6, 5),
                ThreeRectangleVerticalFeature(12, 4, 6, 5),
                TwoRectangleVerticalFeature(3, 4, 14, 15),
                ThreeRectangleHorizontalFeature(3, 4, 14, 15),
                ThreeRectangleHorizontalFeature(12, 3, 5, 15),
                TwoRectangleVerticalFeature(15, 4, 4, 7),
                TwoRectangleVerticalFeature(12, 3, 6, 13),
                ThreeRectangleVerticalFeature(12, 3, 6, 13),
                TwoRectangleVerticalFeature(8, 13, 8, 5),
                TwoRectangleVerticalFeature(5, 3, 12, 15),
                TwoRectangleHorizontalFeature(8, 14, 5, 4),
                ThreeRectangleVerticalFeature(5, 3, 12, 15),
                ThreeRectangleHorizontalFeature(5, 3, 12, 15),
                TwoRectangleHorizontalFeature(3, 4, 13, 14),
                TwoRectangleVerticalFeature(10, 13, 4, 6),
                TwoRectangleVerticalFeature(5, 12, 12, 4),
                TwoRectangleHorizontalFeature(10, 13, 4, 6),
                TwoRectangleHorizontalFeature(5, 12, 12, 4),
                ThreeRectangleVerticalFeature(5, 12, 12, 4),
                ThreeRectangleHorizontalFeature(10, 13, 4, 6),
                FourRectangleFeature(10, 13, 4, 6),
                FourRectangleFeature(5, 12, 12, 4),
                TwoRectangleVerticalFeature(6, 5, 12, 14),
                TwoRectangleHorizontalFeature(6, 5, 12, 14),
                ThreeRectangleVerticalFeature(6, 5, 12, 14),
                FourRectangleFeature(6, 5, 12, 14),
                TwoRectangleVerticalFeature(6, 13, 10, 6),
                TwoRectangleHorizontalFeature(6, 13, 10, 6),
                ThreeRectangleHorizontalFeature(6, 13, 10, 6),
                FourRectangleFeature(6, 13, 10, 6),
                TwoRectangleVerticalFeature(3, 14, 12, 5),
                ThreeRectangleVerticalFeature(3, 14, 12, 5),
                TwoRectangleVerticalFeature(14, 4, 4, 12),
                TwoRectangleHorizontalFeature(14, 4, 4, 12),
                ThreeRectangleHorizontalFeature(14, 4, 4, 12),
                FourRectangleFeature(14, 4, 4, 12),
                TwoRectangleVerticalFeature(12, 3, 6, 14),
                TwoRectangleVerticalFeature(7, 3, 12, 15),
                TwoRectangleHorizontalFeature(12, 3, 6, 14),
                TwoRectangleHorizontalFeature(6, 3, 11, 14),
                ThreeRectangleVerticalFeature(12, 3, 6, 14),
                ThreeRectangleVerticalFeature(7, 3, 12, 15),
                ThreeRectangleHorizontalFeature(7, 3, 12, 15),
                FourRectangleFeature(12, 3, 6, 14),
                TwoRectangleHorizontalFeature(7, 14, 9, 4),
                ThreeRectangleVerticalFeature(7, 14, 9, 4),
                TwoRectangleVerticalFeature(3, 13, 8, 5),
                TwoRectangleVerticalFeature(8, 12, 8, 7),
                ThreeRectangleVerticalFeature(6, 12, 9, 7),
                ThreeRectangleVerticalFeature(4, 14, 9, 5),
                TwoRectangleVerticalFeature(5, 3, 14, 10),
                TwoRectangleHorizontalFeature(5, 3, 14, 10),
                FourRectangleFeature(5, 3, 14, 10),
                TwoRectangleVerticalFeature(11, 3, 6, 13),
                ThreeRectangleVerticalFeature(11, 3, 6, 13),
                TwoRectangleVerticalFeature(14, 3, 4, 15),
                TwoRectangleVerticalFeature(12, 3, 6, 10),
                TwoRectangleHorizontalFeature(12, 3, 6, 10),
                ThreeRectangleVerticalFeature(12, 3, 6, 10),
                ThreeRectangleHorizontalFeature(14, 3, 4, 15),
                FourRectangleFeature(12, 3, 6, 10),
                ThreeRectangleHorizontalFeature(3, 3, 13, 15),
                TwoRectangleVerticalFeature(11, 3, 6, 15),
                ThreeRectangleVerticalFeature(11, 3, 6, 15),
                ThreeRectangleVerticalFeature(9, 4, 9, 15),
                ThreeRectangleHorizontalFeature(11, 3, 6, 15),
                ThreeRectangleHorizontalFeature(9, 4, 9, 15),
                TwoRectangleVerticalFeature(7, 12, 6, 6),
                TwoRectangleHorizontalFeature(7, 12, 6, 6),
                ThreeRectangleVerticalFeature(7, 12, 6, 6),
                ThreeRectangleHorizontalFeature(7, 12, 6, 6),
                FourRectangleFeature(7, 12, 6, 6),
                TwoRectangleVerticalFeature(7, 14, 4, 4),
                TwoRectangleVerticalFeature(6, 12, 10, 6),
                TwoRectangleHorizontalFeature(7, 14, 4, 4),
                TwoRectangleHorizontalFeature(6, 12, 10, 6),
                TwoRectangleHorizontalFeature(3, 5, 11, 14),
                ThreeRectangleHorizontalFeature(6, 12, 10, 6),
                FourRectangleFeature(7, 14, 4, 4),
                FourRectangleFeature(6, 12, 10, 6),
                TwoRectangleVerticalFeature(4, 5, 12, 14),
                TwoRectangleHorizontalFeature(4, 5, 12, 14),
                ThreeRectangleVerticalFeature(4, 5, 12, 14),
                FourRectangleFeature(4, 5, 12, 14),
                TwoRectangleHorizontalFeature(6, 12, 11, 4),
                TwoRectangleVerticalFeature(5, 5, 4, 4),
                TwoRectangleHorizontalFeature(5, 5, 4, 4),
                FourRectangleFeature(5, 5, 4, 4),
                TwoRectangleHorizontalFeature(7, 11, 5, 8),
                TwoRectangleVerticalFeature(5, 12, 12, 7),
                ThreeRectangleVerticalFeature(5, 12, 12, 7),
                TwoRectangleHorizontalFeature(7, 3, 7, 14),
                TwoRectangleVerticalFeature(14, 4, 4, 11),
                TwoRectangleHorizontalFeature(10, 4, 9, 10),
                TwoRectangleHorizontalFeature(8, 3, 11, 10),
                ThreeRectangleVerticalFeature(10, 4, 9, 10),
                ThreeRectangleVerticalFeature(4, 4, 15, 5),
                TwoRectangleHorizontalFeature(14, 5, 5, 4),
                TwoRectangleVerticalFeature(8, 14, 4, 4),
                TwoRectangleHorizontalFeature(8, 14, 4, 4),
                TwoRectangleHorizontalFeature(4, 4, 13, 14),
                FourRectangleFeature(8, 14, 4, 4),
                TwoRectangleVerticalFeature(6, 12, 6, 5),
                TwoRectangleHorizontalFeature(7, 11, 5, 6),
                ThreeRectangleVerticalFeature(6, 12, 6, 5),
                ThreeRectangleHorizontalFeature(7, 11, 5, 6),
                TwoRectangleVerticalFeature(5, 5, 4, 14),
                TwoRectangleVerticalFeature(9, 13, 6, 6),
                TwoRectangleHorizontalFeature(5, 5, 4, 14),
                TwoRectangleHorizontalFeature(9, 13, 6, 6),
                TwoRectangleHorizontalFeature(5, 5, 11, 14),
                ThreeRectangleVerticalFeature(9, 13, 6, 6),
                ThreeRectangleHorizontalFeature(9, 13, 6, 6),
                FourRectangleFeature(5, 5, 4, 14),
                FourRectangleFeature(9, 13, 6, 6),
                TwoRectangleVerticalFeature(6, 4, 12, 15),
                ThreeRectangleVerticalFeature(6, 4, 12, 15),
                ThreeRectangleHorizontalFeature(6, 4, 12, 15),
                TwoRectangleHorizontalFeature(9, 13, 7, 6),
                ThreeRectangleHorizontalFeature(9, 13, 7, 6),
                ThreeRectangleVerticalFeature(3, 14, 9, 5),
                TwoRectangleVerticalFeature(13, 3, 6, 15),
                ThreeRectangleVerticalFeature(13, 3, 6, 15),
                ThreeRectangleHorizontalFeature(13, 3, 6, 15),
                ThreeRectangleVerticalFeature(4, 3, 9, 15),
                ThreeRectangleHorizontalFeature(4, 3, 9, 15),
                TwoRectangleVerticalFeature(10, 4, 8, 12),
                TwoRectangleVerticalFeature(11, 3, 8, 13),
                TwoRectangleVerticalFeature(7, 3, 10, 13),
                TwoRectangleHorizontalFeature(11, 4, 7, 10),
                TwoRectangleHorizontalFeature(10, 4, 8, 12),
                ThreeRectangleHorizontalFeature(10, 4, 8, 12),
                FourRectangleFeature(10, 4, 8, 12),
                TwoRectangleHorizontalFeature(7, 4, 11, 12),
                ThreeRectangleHorizontalFeature(7, 4, 11, 12),
                TwoRectangleVerticalFeature(10, 5, 8, 4),
                TwoRectangleHorizontalFeature(10, 5, 8, 4),
                TwoRectangleHorizontalFeature(6, 12, 11, 6),
                ThreeRectangleHorizontalFeature(6, 12, 11, 6),
                FourRectangleFeature(10, 5, 8, 4),
                TwoRectangleVerticalFeature(4, 15, 12, 4),
                TwoRectangleHorizontalFeature(4, 15, 12, 4),
                ThreeRectangleVerticalFeature(4, 15, 12, 4),
                FourRectangleFeature(4, 15, 12, 4),
                ThreeRectangleVerticalFeature(9, 3, 9, 15),
                ThreeRectangleHorizontalFeature(9, 3, 9, 15),
                TwoRectangleVerticalFeature(11, 13, 4, 6),
                TwoRectangleVerticalFeature(4, 3, 14, 14),
                TwoRectangleHorizontalFeature(11, 13, 4, 6),
                TwoRectangleHorizontalFeature(4, 3, 14, 14),
                ThreeRectangleVerticalFeature(10, 4, 9, 5),
                ThreeRectangleHorizontalFeature(11, 13, 4, 6),
                FourRectangleFeature(11, 13, 4, 6),
                FourRectangleFeature(4, 3, 14, 14),
                TwoRectangleVerticalFeature(10, 4, 8, 14),
                TwoRectangleHorizontalFeature(10, 4, 8, 14),
                FourRectangleFeature(10, 4, 8, 14),
                ThreeRectangleVerticalFeature(6, 4, 9, 15),
                ThreeRectangleHorizontalFeature(6, 4, 9, 15),
                TwoRectangleVerticalFeature(8, 13, 6, 6),
                TwoRectangleVerticalFeature(5, 5, 12, 14),
                TwoRectangleHorizontalFeature(8, 13, 6, 6),
                TwoRectangleHorizontalFeature(5, 5, 12, 14),
                TwoRectangleHorizontalFeature(3, 5, 15, 4),
                ThreeRectangleVerticalFeature(8, 13, 6, 6),
                ThreeRectangleVerticalFeature(5, 5, 12, 14),
                ThreeRectangleVerticalFeature(3, 5, 15, 4),
                ThreeRectangleHorizontalFeature(8, 13, 6, 6),
                FourRectangleFeature(8, 13, 6, 6),
                FourRectangleFeature(5, 5, 12, 14),
                ThreeRectangleHorizontalFeature(5, 4, 13, 15),
                TwoRectangleVerticalFeature(11, 12, 4, 7),
                TwoRectangleVerticalFeature(14, 4, 4, 9),
                TwoRectangleVerticalFeature(10, 14, 6, 4),
                TwoRectangleVerticalFeature(8, 12, 6, 7),
                TwoRectangleVerticalFeature(7, 13, 8, 6),
                TwoRectangleVerticalFeature(6, 5, 10, 14),
                TwoRectangleVerticalFeature(4, 14, 12, 5),
                TwoRectangleVerticalFeature(4, 3, 12, 15),
                TwoRectangleVerticalFeature(4, 4, 14, 15),
                TwoRectangleHorizontalFeature(7, 12, 5, 4),
                TwoRectangleHorizontalFeature(10, 14, 6, 4),
                TwoRectangleHorizontalFeature(9, 14, 7, 4),
                TwoRectangleHorizontalFeature(7, 13, 8, 6),
                TwoRectangleHorizontalFeature(7, 12, 9, 6),
                TwoRectangleHorizontalFeature(6, 5, 10, 14),
                TwoRectangleHorizontalFeature(6, 3, 13, 10),
                ThreeRectangleVerticalFeature(10, 14, 6, 4),
                ThreeRectangleVerticalFeature(8, 12, 6, 7),
                ThreeRectangleVerticalFeature(7, 12, 9, 6),
                ThreeRectangleVerticalFeature(4, 14, 12, 5),
                ThreeRectangleVerticalFeature(4, 3, 12, 15),
                ThreeRectangleVerticalFeature(3, 4, 15, 13),
                ThreeRectangleVerticalFeature(3, 4, 15, 15),
                ThreeRectangleHorizontalFeature(14, 4, 4, 9),
                ThreeRectangleHorizontalFeature(7, 13, 8, 6),
                ThreeRectangleHorizontalFeature(7, 12, 9, 6),
                ThreeRectangleHorizontalFeature(4, 3, 12, 15),
                ThreeRectangleHorizontalFeature(4, 4, 14, 15),
                ThreeRectangleHorizontalFeature(3, 4, 15, 15),
                FourRectangleFeature(10, 14, 6, 4),
                FourRectangleFeature(7, 13, 8, 6),
                FourRectangleFeature(6, 5, 10, 14), ]
    else:
        raise ValueError('Unknown type:', type)
    return features
