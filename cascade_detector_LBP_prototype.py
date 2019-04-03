
import numpy as np
from operator import itemgetter
from helper import load_features, load_training_data, load_test_data
from integral_image import *

from LBP_image import *

class WeakClassifier(object):

    def __init__(self, feature_num, polarity, threshold):
        self.feature_num = feature_num # Take the n first features
        self.polarity = polarity
        self.threshold = threshold

    def __call__(self, *args, **kwargs):
        proc_image = args[0]

        if self.polarity == 1:
            comp = lambda x: 1 if x < self.threshold else 0
        else:
            comp = lambda x: 1 if x > self.threshold else 0
        #
        f_x = proc_image[:,self.feature_num]
        f_x = f_x > self.threshold
        return f_x.astype(int)


class StrongClassifier(object):

    def __init__(self, weak_classifiers, alphas, threshold=None):
        self.weak_classifiers = weak_classifiers
        self.alphas = alphas
        self.r = threshold if threshold else np.sum(alphas) / 2.0

    def predict(self, X):
        l = np.sum([alpha * h(X)
                    for h, alpha in zip(self.weak_classifiers, self.alphas)], axis=0)
        return np.where(l >= self.r, 1, 0)


class AdaBoost(object):

    def __init__(self, verbose=False):
        self.verbose = verbose

    def initial_weights(self, Y):
        m, l = 0.0, 0.0
        for y in Y:
            if y == 1:
                m += 1.0
            else:
                l += 1.0

        w0_p, w0_n = (1.0 / (2 * m)), (1.0 / (2 * l))
        return [w0_p if y == 1 else w0_n for y in Y]

    def find_min_error(self, f_x, Y, weights):
        '''
        Find the minimum weighted error for the feature over given example images.

        References
            Viola, Paul, and Michael J. Jones. "Robust real-time face detection."
            International journal of computer vision 57.2 (2004): 137-154.

        :param f_x: The list of feature values
        :param Y: Given example labels
        :param weights: Normalize the weights at t
        :return: The optimal threshold with polarity and error
        '''
        T_plus, T_minus = 0.0, 0.0
        for y, weight in zip(Y, weights):
            if y == 1:
                T_plus += weight
            else:
                T_minus += weight

        res, e_t = None, float("inf")
        for f_i, f in enumerate(f_x):
            f = np.asarray([(f_sum, y, weight)
                            for f_sum, y, weight in zip(f, Y, weights)],
                           dtype=[('value', np.float64), ('y', int), ('weight', float)])
            f = np.sort(f, order='value')

            errors = []
            S_plus, S_minus = T_plus, T_minus
            for threshold, y, weight in reversed(f):
                if y == 1:
                    S_plus -= weight
                else:
                    S_minus -= weight

                l = S_plus + T_minus - S_minus
                r = S_minus + T_plus - S_plus
                e = min(l, r)
                errors.append((e, 1 if e == r else -1, threshold))

            e, p, t = min(errors, key=itemgetter(0))
            if e < e_t:
                e_t = e
                res = (f_i, e, p, t)
        return res

    def boost(self, X, Y, num, iter=100):
        w_0 = self.initial_weights(Y)
        weights = [w_0, ]
        weak_classifiers = []
        errors = []
        alphas = []
        # 0. Initialize
        f_x = X[:,0:num]
        f_x = f_x.T
        if self.verbose:
            print('Pre-calculated the sum of integral images over %d features.' % (num))

        for t in range(iter):
            # 1. Normalize the weights
            weights[-1] = np.asarray(weights[-1], dtype=np.float64) / \
                          np.sum(weights[-1])

            # 2. For each feature, j, train a classifier hj which is restricted to using a single feature.
            w_t = weights[-1]
            assert np.isclose(1.0, np.sum(w_t))

            # 3. Choose the classifier, ht, with the lowest error et.
            f_i, e_t, p, theta = self.find_min_error(f_x, Y, w_t)
            if self.verbose:
                print('[Iteration %d] found optimal threshold %f for ' \
                      'the feature %d with the polarity %d and the error %f' % ((t + 1), theta, f_i, p, e_t))
            h_t = WeakClassifier(num, p, theta)

            weak_classifiers.append(h_t)
            errors.append(e_t)

            # 4. Update the weights
            beta = e_t / (1 - e_t)
            alphas.append(np.log(1 / beta))

            next_weights = w_t * np.power(beta, 1.0 - np.abs(h_t(X) - Y))
            weights.append(next_weights)

        return weak_classifiers, alphas


class CascadeClassifier(object):

    def __init__(self, f=.5, d=.9, F_thresh=.03125, verbose=True):
        self.f = f  # Cascade structure results in false positive rate of about 10^-6
        self.d = d  # Results in detection rate of 90 percent
        self.F_thresh = F_thresh
        self.classifiers = []
        self.verbose = verbose

        self.boosting_algorithm = AdaBoost(verbose=verbose)
        self.T = 100  # iteration for boosting

    def split(self, X, Y, dev_prop=.1, shuffle=True):
        # Shrink dataset for faster testing
        data = [(x, y) for x, y in zip(X.pi, Y)]
        if shuffle:
            np.random.shuffle(data)

        idx = int(len(data) * (1.0 - dev_prop))
        train = data[:idx]
        dev = data[idx:]

        X_train, Y_train = [], []
        for x, y in train:
            X_train.append(x)
            Y_train.append(y)

        X_dev, Y_dev = [], []
        for x, y in dev:
            X_dev.append(x)
            Y_dev.append(y)

        return np.array(X_train), np.array(Y_train), np.array(X_dev), np.array(Y_dev)

    def get_positive_negative_set(self, X, Y):
        PX, PY, NX, NY = [], [], [], []
        for x, y in zip(X, Y):
            if y == 1:
                PX.append(x)
                PY.append(y)
            else:
                NX.append(x)
                NY.append(y)
        return np.asarray(PX), np.asarray(PY), np.asarray(NX), np.asarray(NY)

    def train(self, X, Y):
        # Use ten 20 feature classifiers

        F, D, n, i = [1, ], [1, ], [0, ], 0
        #features = load_features(self.feature_type)

        X_train, Y_train, X_dev, Y_dev = self.split(X, Y)
        PX, PY, NX, NY = self.get_positive_negative_set(X_train, Y_train)
        # Integrate PX and NX
        #PX, NX = IntegralImage(PX), IntegralImage(NX)
        #X_dev = IntegralImage(X_dev)

        c_list = []
        y_pred = [1, ]

        while F[i] > self.F_thresh and sum(y_pred) != 0:
            i += 1
            n.append(n[i - 1])
            D.append(0)
            F.append(F[i - 1])
            inc = 5

            while F[i] > self.f * F[i - 1]:
                n[i] += inc

                # Use P and N to train classifier
                X_train = np.concatenate((PX, NX))
                Y_train = np.concatenate((PY, NY))

                weak_classifiers, alphas = self.boosting_algorithm.boost(
                        X_train, Y_train, num=n[i], iter=self.T)
                #weak_classifiers, alphas = self.boosting_algorithm.boost(
                #    X_train, Y_train, self.select_features(features, stage=i, num=n[i]), iter=self.T)
                threshold = sum(alphas) / 2.0
                step = threshold / 20.0
                D[i] = 0

                # Evaluate current cascaded classifier on validation set to determine Fi and Di.
                # Loop until Di < d * D[i-1]. Decrease threshold by 1 everytime until detection rate is greater
                while D[i] < self.d * D[i - 1]:
                    c = StrongClassifier(weak_classifiers, alphas, threshold)

                    TP, FP, TN, FN = self.evaluate(c_list + [c, ], X_dev, Y_dev)
                    F[i] = FP / float(FP + TN)
                    D[i] = TP / float(FN + TP)

                    if self.verbose:
                        print("F", F[i], "D", D[i])

                    if D[i] < self.d * D[i - 1]:
                        # Decrement threshold by one step
                        threshold -= step
                        if threshold == 0:
                            threshold = 1
                inc += 2

            y_pred = c.predict(NX)
            NX.pi = NX.pi[y_pred == 1]
            NY = NY[y_pred == 1]
            c_list.append(c)

        # Print results
        if self.verbose:
            print("Detection Rates:", D)
            print("False Positive Rates:", F)
            print("Number of Features per Cascade", n[1:5])

        self.classifiers = c_list

    def train_forced(self, X, Y, n):
        # Train algorithm used fixed parameters for number of cascades and other such things

        F, D = [], []
        # features = load_features(self.feature_type)
        X_train, Y_train, X_dev, Y_dev = self.split(X, Y)
        PX, PY, NX, NY = self.get_positive_negative_set(X_train, Y_train)

        c_list = []
        for i in range(4):
            # Use P and N to train classifier
            X_train = np.concatenate((PX, NX))
            Y_train = np.concatenate((PY, NY))
            weak_classifiers, alphas = self.boosting_algorithm.boost(
                X_train, Y_train, num=n[i], iter=self.T)
            threshold = sum(alphas) / 2.0
            # Get F[i] and D[i] for current cascade
            # Evaluate current cascaded classifier on validation set to determine Fi and Di.
            # Loop until Di < d * D[i-1]. Decrease threshold by 1 everytime until detection rate is greater
            c = StrongClassifier(weak_classifiers, alphas, threshold)
            TP, FP, TN, FN = self.evaluate(c_list + [c, ], X_dev, Y_dev)
            F.append(FP / float(FP + TN))
            D.append(TP / float(FN + TP))

            y_pred = c.predict(NX)
            NX = NX[y_pred == 1]
            NY = NY[y_pred == 1]
            c_list.append(c)

        # Print results
        if self.verbose:
            print("Detection Rates:", D)
            print("False Positive Rates:", F)
            print("Number of Features per Cascade", n)

        self.classifiers = c_list
    def evaluate(self, c_list, X, Y):
        FN, TN = 0, 0
        Xval_tmp = X
        for c in c_list:
            y_pred = c.predict(Xval_tmp)
            y_true = Y

            from sklearn.metrics import precision_score
            print (precision_score(y_true, y_pred))

            # Count number of false negatives and true negatives
            TN += sum(np.logical_and(y_pred == 0, y_true == 0))
            FN += sum(np.logical_and(y_pred == 0, y_true == 1))

            # If take positive examples and continue running
            Xval_tmp = Xval_tmp[y_pred == 1]
            Y = Y[y_pred == 1]

        # Count number of true positives and false positives
        FP = sum(np.logical_and(y_pred == 1, y_true == 0))
        TP = sum(np.logical_and(y_pred == 1, y_true == 1))

        if self.verbose:
            print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)

        return TP, FP, TN, FN

    def predict(self, X):
        y_pred = self.classifiers[0].predict(X)
        for classifier in self.classifiers[1:]:
            X_indices = [i for i, y in zip(range(len(X)), y_pred) if y == 1]
            subX = X[X_indices]
            Y_hat = classifier.predict(subX)
            X_indices = []

            for i, y in zip(X_indices, Y_hat):
                if y == 0:
                    y_pred[i] = 0

        return y_pred


if __name__ == '__main__':
    from sklearn.metrics import precision_score

    dataset = 'colorferet'
    X_train, Y_train, X_test_pos, Y_test_pos  = load_training_data(dataset=dataset)
    X_train = ProcessedImage(X_train)
    cascade_classifier = CascadeClassifier(verbose=True)
    cascade_classifier.train_forced(X_train, Y_train,[17, 26, 37, 101])
    del X_train, Y_train
    X_test, Y_test = load_test_data(dataset=dataset)
    if dataset == 'colorferet':
        X_test = np.concatenate((X_test,X_test_pos))
        Y_test = np.concatenate((Y_test, Y_test_pos))
    X_test = ProcessedImage(X_test).pi
    Y_pred = cascade_classifier.predict(X_test)
    FP = sum(np.logical_and(Y_pred == 1, Y_test == 0))
    TP = sum(np.logical_and(Y_pred == 1, Y_test == 1))
    TN = sum(np.logical_and(Y_pred == 0, Y_test == 0))
    FN = sum(np.logical_and(Y_pred == 0, Y_test == 1))
    print("Test Evaluation")
    print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)
    print(precision_score(y_true=Y_test, y_pred=Y_pred))
