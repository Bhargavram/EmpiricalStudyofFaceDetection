from abc import abstractmethod, ABCMeta
from itertools import product

import numpy as np
from matplotlib.patches import Rectangle


class Feature(object):
    __metaclass__ = ABCMeta

    def __init__(self, x, y, w, h):
        if x < 0:
            raise ValueError('The argument x should be greater than zero.')
        if y < 0:
            raise ValueError('The argument y should be greater than zero.')
        if w < 1:
            raise ValueError('The argument w should be greater than one.')
        if h < 1:
            raise ValueError('The argument h should be greater than one.')

        self.x, self.y = x, y
        self.width, self.height = w, h

    @abstractmethod
    def area(self, ii):
        pass

    def __str__(self):
        return '{name} {w}x{h}: ({x}, {y})' \
            .format(name=type(self).__name__,
                    w=self.width,
                    h=self.height,
                    x=self.x, y=self.y)


class TwoRectangleVerticalFeature(Feature):

    @classmethod
    def window(cls):
        return 2, 1

    @classmethod
    def is_fit(cls, w, h):
        return (w % 2 == 0) and (h > 0)

    def area(self, ii):
        '''
        1---2---3
        | o | x |
        4---5---6

        Calculate (x - o),

        x = 6 + 2 - (3 + 5)
        o = 5 + 1 - (2 + 4)

        :param ii: an instance of IntegralImage
        :return:
        '''

        w = (self.width / 2)

        x = ii.sum_rectangle(x=self.x + w,
                             y=self.y,
                             w=w,
                             h=self.height)
        o = ii.sum_rectangle(x=self.x,
                             y=self.y,
                             w=w,
                             h=self.height)

        return x - o

    def rectangles(self):
        w = self.width / 2
        return [
            Rectangle((self.x, self.y), w, self.height,
                      linewidth=1, edgecolor='r', facecolor='white'),
            Rectangle((self.x + w, self.y), w, self.height,
                      linewidth=1, edgecolor='r', facecolor='black')
        ]


class TwoRectangleHorizontalFeature(Feature):

    @classmethod
    def window(cls):
        return 1, 2

    @classmethod
    def is_fit(cls, w, h):
        return (w > 0) and (h % 2 == 0)

    def area(self, ii):
        '''
        1---2
        | x |
        3---4
        | o |
        5---6

        Calculate (x - o),

        x = 4 + 1 - (2 + 3)
        o = 6 + 3 - (4 + 5)

        :param ii: an instance of IntegralImage
        :return:
        '''
        h = self.height / 2

        x = ii.sum_rectangle(x=self.x,
                             y=self.y,
                             w=self.width,
                             h=h)
        o = ii.sum_rectangle(x=self.x,
                             y=self.y + h,
                             w=self.width,
                             h=h)

        return x - o

    def rectangles(self):
        h = self.height / 2
        return [
            Rectangle((self.x, self.y), self.width, h,
                      linewidth=1, edgecolor='r', facecolor='black'),
            Rectangle((self.x, self.y + h), self.width, h,
                      linewidth=1, edgecolor='r', facecolor='white')
        ]


class ThreeRectangleVerticalFeature(Feature):

    @classmethod
    def window(cls):
        return 3, 1

    @classmethod
    def is_fit(cls, w, h):
        return (w % 3 == 0) and (h > 0)

    def area(self, ii):
        '''
        1---2---3---4
        | o | x | O |
        5---6---7---8

        Calculate (x - o - O),

        x = 7 + 2 - (3 + 6)
        o = 6 + 1 - (2 + 5)
        O = 8 + 3 - (4 + 7)

        :param ii: an instance of IntegralImage
        :return:
        '''

        w = (self.width / 3)
        x = ii.sum_rectangle(x=self.x + w,
                             y=self.y,
                             w=w,
                             h=self.height)
        o = ii.sum_rectangle(x=self.x,
                             y=self.y,
                             w=w,
                             h=self.height)
        O = ii.sum_rectangle(x=self.x + (w * 2),
                             y=self.y,
                             w=w,
                             h=self.height)

        return x - o - O

    def rectangles(self):
        w = (self.width / 3)
        return [
            Rectangle((self.x, self.y), w, self.height,
                      linewidth=1, edgecolor='r', facecolor='white'),
            Rectangle((self.x + w, self.y), w, self.height,
                      linewidth=1, edgecolor='r', facecolor='black'),
            Rectangle((self.x + (w * 2), self.y), w, self.height,
                      linewidth=1, edgecolor='r', facecolor='white')
        ]


class ThreeRectangleHorizontalFeature(Feature):

    @classmethod
    def window(cls):
        return 1, 3

    @classmethod
    def is_fit(cls, w, h):
        return (w > 0) and (h % 3 == 0)

    def area(self, ii):
        '''
        1---2
        | o |
        3---4
        | x |
        5---6
        | O |
        7---8

        Calculate (x - o - O),

        x = 6 + 3 - (4 + 5)
        o = 4 + 1 - (2 + 3)
        O = 8 + 5 - (6 + 7)

        :param ii: an instance of IntegralImage
        :return:
        '''

        h = (self.height / 3)
        x = ii.sum_rectangle(x=self.x,
                             y=self.y + h,
                             w=self.width,
                             h=h)
        o = ii.sum_rectangle(x=self.x,
                             y=self.y,
                             w=self.width,
                             h=h)
        O = ii.sum_rectangle(x=self.x,
                             y=self.y + (h * 2),
                             w=self.width,
                             h=h)

        return x - o - O

    def rectangles(self):
        h = self.height / 3
        return [
            Rectangle((self.x, self.y), self.width, h,
                      linewidth=1, edgecolor='r', facecolor='white'),
            Rectangle((self.x, self.y + h), self.width, h,
                      linewidth=1, edgecolor='r', facecolor='black'),
            Rectangle((self.x, self.y + (h * 2)), self.width, h,
                      linewidth=1, edgecolor='r', facecolor='white')
        ]


class ThreeExtendedRectangleVerticalFeature(Feature):

    @classmethod
    def window(cls):
        return 4, 1

    @classmethod
    def is_fit(cls, w, h):
        return (w % 4 == 0) and (h > 0)

    def area(self, ii):
        '''
        1---2---+---3---4
        | o | x | x | O |
        5---6---+---7---8

        Calculate x - (o1+o2),

        x = 7 + 2 - (3 + 6)
        o = 6 + 1 - (2 + 5)
        O = 8 + 3 - (4 + 7)

        :param ii: an instance of IntegralImage
        :return:
        '''

        w = (self.width / 4)
        x = ii.sum_rectangle(x=self.x + w,
                             y=self.y,
                             w=(w * 2),
                             h=self.height)
        o = ii.sum_rectangle(x=self.x,
                             y=self.y,
                             w=w,
                             h=self.height)
        O = ii.sum_rectangle(x=self.x + (w * 3),
                             y=self.y,
                             w=w,
                             h=self.height)

        return x - o - O

    def rectangles(self):
        w = (self.width / 4)
        return [
            Rectangle((self.x + w, self.y), (w * 2), self.height,
                      linewidth=1, edgecolor='r', facecolor='black'),
            Rectangle((self.x, self.y), w, self.height,
                      linewidth=1, edgecolor='r', facecolor='white'),
            Rectangle((self.x + (w * 3), self.y), w, self.height,
                      linewidth=1, edgecolor='r', facecolor='white'),
        ]


class ThreeExtendedRectangleHorizontalFeature(Feature):

    @classmethod
    def window(cls):
        return 1, 4

    @classmethod
    def is_fit(cls, w, h):
        return (w > 0) and (h % 4 == 0)

    def area(self, ii):
        '''
        1---2
        | o |
        3---4
        | x |
        +---+
        | x |
        5---6
        | O |
        7---8

        Calculate (x - o - O),

        x = 6 + 3 - (4 + 5)
        o = 4 + 1 - (2 + 3)
        O = 8 + 5 - (6 + 7)

        :param ii: an instance of IntegralImage
        :return:
        '''

        h = (self.height / 4)
        x = ii.sum_rectangle(x=self.x,
                             y=self.y + h,
                             w=self.width,
                             h=(h * 2))
        o = ii.sum_rectangle(x=self.x,
                             y=self.y,
                             w=self.width,
                             h=h)
        O = ii.sum_rectangle(x=self.x,
                             y=self.y + (h * 3),
                             w=self.width,
                             h=h)

        return x - o - O

    def rectangles(self):
        h = (self.height / 4)
        return [
            Rectangle((self.x, self.y + h), self.width, (h * 2),
                      linewidth=1, edgecolor='r', facecolor='black'),
            Rectangle((self.x, self.y), self.width, h,
                      linewidth=1, edgecolor='r', facecolor='white'),
            Rectangle((self.x, self.y + (h * 3)), self.width, h,
                      linewidth=1, edgecolor='r', facecolor='white'),
        ]


class FourRectangleFeature(Feature):

    @classmethod
    def window(cls):
        return 2, 2

    @classmethod
    def is_fit(cls, w, h):
        return (w % 2 == 0) and (h % 2 == 0)

    def area(self, ii):
        '''
        1---2---3
        | o | x |
        4---5---6
        | X | O |
        7---8---9

        Calculate (x + X - o - O),

        x = 6 + 2 - (3 + 5)
        X = 8 + 4 - (5 + 7)
        o = 5 + 1 - (2 + 4)
        O = 9 + 5 - (6 + 8)

        :param ii: an instance of IntegralImage
        :return:
        '''

        w = (self.width / 2)
        h = (self.height / 2)
        x = ii.sum_rectangle(x=self.x + w,
                             y=self.y,
                             w=w,
                             h=h)
        X = ii.sum_rectangle(x=self.x,
                             y=self.y + h,
                             w=w,
                             h=h)
        o = ii.sum_rectangle(x=self.x,
                             y=self.y,
                             w=w,
                             h=h)
        O = ii.sum_rectangle(x=self.x + w,
                             y=self.y + h,
                             w=w,
                             h=h)

        return x + X - o - O

    def rectangles(self):
        w = (self.width / 2)
        h = (self.height / 2)
        return [
            Rectangle((self.x + w, self.y), w, h,
                      linewidth=1, edgecolor='r', facecolor='black'),
            Rectangle((self.x, self.y + h), w, h,
                      linewidth=1, edgecolor='r', facecolor='black'),
            Rectangle((self.x, self.y), w, h,
                      linewidth=1, edgecolor='r', facecolor='white'),
            Rectangle((self.x + w, self.y + h), w, h,
                      linewidth=1, edgecolor='r', facecolor='white'),
        ]


class CenterSurroundedRectangleFeature(Feature):

    @classmethod
    def window(cls):
        return 3, 3

    @classmethod
    def is_fit(cls, w, h):
        return (w % 3 == 0) and (h % 3 == 0)

    def area(self, ii):
        '''
        1---2---3---4
        | a | b | c |
        5---6---7---8
        | d | x | e |
        9--10--11--12
        | f | g | h |
       13--14--15--16

        Calculate x - (a+b+c+d+e+f+g) => x - (16 - x) = -16 + 2x

        :param ii: an instance of IntegralImage
        :return:
        '''

        w = (self.width / 3)
        h = (self.height / 3)
        x = ii.sum_rectangle(x=self.x + w,
                             y=self.y + h,
                             w=w,
                             h=h)
        s = ii.sum_rectangle(x=self.x,
                             y=self.y,
                             w=self.width,
                             h=self.height)

        return -s + (2 * x)

    def rectangles(self):
        w = (self.width / 3)
        h = (self.height / 3)
        return [
            Rectangle((self.x, self.y), self.width, self.height,
                      linewidth=1, edgecolor='r', facecolor='white'),
            Rectangle((self.x + w, self.y + h), w, h,
                      linewidth=1, edgecolor='r', facecolor='black'),
        ]


class IntegralImage(object):

    def __init__(self, images):
        if isinstance(images, np.ndarray) and len(images.shape) == 2:
            images = np.asarray([images, ])

        if len(images) > 0:
            self.ii = np.stack([self.__generate_sat(image) for image in images])
        else:
            self.ii = None

    @staticmethod
    def from_array(arr):
        obj = IntegralImage([])
        obj.ii = arr
        return obj

    def __generate_sat(self, i):
        shape = i.shape
        row, column = shape[0], shape[1]
        _sat = np.zeros(shape)

        def sat(x0, y0):
            if -1 < x0 < row and -1 < y0 < column:
                return _sat[x0, y0]
            return 0.0

        for y, x in product(range(row), range(column)):
            _sat[y, x] = sat(y - 1, x) + sat(y, x - 1) + i[y, x] - sat(y - 1, x - 1)

        return _sat

    def __getitem__(self, item):
        x, y = item[0], item[1]
        if x < 0 or y < 0:
            return np.zeros(len(self.ii))

        return self.ii[:, x, y]

    @classmethod
    def generate_features(cls, shape, feature_class):
        if not issubclass(feature_class, Feature):
            raise ValueError

        width, height = shape
        features = []
        X, Y = feature_class.window()

        for w, h in product(range(X, width + 1, X), range(Y, height + 1, Y)):
            if not feature_class.is_fit(w, h):
                continue

            for x, y in product(range(0, width - (w - 1)), range(0, height - (h - 1))):
                feature = feature_class(x, y, w, h)
                features.append(feature)
        return features

    def sum(self, feature):
        if not isinstance(feature, Feature):
            raise ValueError

        return self.sum_rectangle(feature.x, feature.y, feature.width, feature.height)

    def sum_rectangle(self, x, y, w, h):
        '''

        :param x: the x-coordinate of the left most corner
        :param y: the y-coordinate of the left most corner
        :param w: the width of rectangle
        :param h: the height of rectangle
        :return: the sum of given rectangle
        '''

        D = self[y + h - 1, x + w - 1]
        A = self[y - 1, x - 1]
        B = self[y + h - 1, x - 1]
        C = self[y - 1, x + w - 1]

        return D + A - B - C


class TiltedIntegralImage(IntegralImage):

    def __init__(self, images):
        if isinstance(images, np.ndarray) and len(images.shape) == 2:
            images = np.asarray([images, ])

        self.ii = np.stack([self.__generate_rsat(image) for image in images])

    def __generate_rsat(self, i):
        shape = i.shape
        row, column = shape[0], shape[1]
        _rsat = np.zeros(shape)

        def rsat(x0, y0):
            if -1 < x0 < row and -1 < y0 < column:
                return _rsat[x0, y0]
            return 0.0

        for y, x in product(range(row), range(column)):
            _rsat[y, x] = rsat(y - 1, x - 1) + rsat(y, x - 1, ) + i[y, x] - rsat(y - 1, x - 2)

        for x, y in product(reversed(range(column)), reversed(range(row))):
            _rsat[y, x] = rsat(y, x) + rsat(y + 1, x - 1) - rsat(y, x - 2)

        return _rsat

    @classmethod
    def generate_features(cls, shape, feature_class):
        if not issubclass(feature_class, Feature):
            raise ValueError

        width, height = shape
        features = []
        X, Y = feature_class.window()

        for w, h in product(range(X, width + 1, X), range(Y, height + 1, Y)):
            if not feature_class.is_fit(w, h):
                continue

            for x, y in product(range(h, (width - 1) - w), range(0, (height - 1) - w - h)):
                feature = feature_class(x, y, w, h)
                features.append(feature)
        return features

    def sum_rectangle(self, x, y, w, h):
        '''

        :param x: the x-coordinate of the left most corner
        :param y: the y-coordinate of the left most corner
        :param w: the width of rectangle
        :param h: the height of rectangle
        :return: the sum of given rectangle
        '''

        D = self[y + w, x + w]
        A = self[y + h, x - h]
        B = self[y, x]
        C = self[y + w + h, x + w - h]

        return D + A - B - C


OriginalFeatures = [
    TwoRectangleVerticalFeature,
    TwoRectangleHorizontalFeature,
    ThreeRectangleVerticalFeature,
    ThreeRectangleHorizontalFeature,
    FourRectangleFeature,
]

ExtendedFeatures = [
    TwoRectangleVerticalFeature,
    TwoRectangleHorizontalFeature,
    ThreeRectangleVerticalFeature,
    ThreeRectangleHorizontalFeature,
    ThreeExtendedRectangleVerticalFeature,
    ThreeExtendedRectangleHorizontalFeature,
    CenterSurroundedRectangleFeature,
]
