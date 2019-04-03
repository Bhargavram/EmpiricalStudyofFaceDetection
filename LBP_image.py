from abc import abstractmethod, ABCMeta
from itertools import product

import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


class Feature(object):
    __metaclass__ = ABCMeta

    def __init__(self, x, y):
        if x < 0:
            raise ValueError('The argument x should be greater than zero.')
        if y < 0:
            raise ValueError('The argument y should be greater than zero.')

        self.x, self.y = x, y

    @abstractmethod
    def area(self, ii):
        pass

    def __str__(self):
        return '{name} {w}x{h}: ({x}, {y})' \
            .format(name=type(self).__name__,
                    w=self.width,
                    h=self.height,
                    x=self.x, y=self.y)


class LBPFeature(Feature):

    def BP(self, image):
        i = 0


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

class ProcessedImage(object):

    def __init__(self, images):
        if isinstance(images, np.ndarray) and len(images.shape) == 2:
            images = np.asarray([images, ])

        if len(images) > 0:
            self.pi = np.stack([self.generate_arr(image) for image in images])
        else:
            self.pi = None

    @staticmethod
    def from_array(arr):
        obj = ProcessedImage([])
        obj.pi = arr
        return obj

    def LBP_get(self,im):
        thresh = im[1,1]
        im = im.reshape(9,)
        im = im[[True,True,True,True,False,True,True,True,True]]
        im = im > thresh
        val = 0
        for i in range(8):
            if im[i]:
                val += (2**i)
        return val
    def generate_arr(self, i):
        # Processes image
        shape = i.shape
        row, column = shape[0], shape[1]
        i = np.pad(i,((1,1),(1,1)),'edge')
        arr = np.zeros([row,column])
        for y in range(column):
            for x in range(row):
                # Get LBP code
                arr[y,x] = self.LBP_get(i[y:y + 3, x:x + 3])
        arr = arr.reshape(row*column)/8
        #np.random.seed(1)
        #arr = np.random.shuffle(arr)
        return arr

    def __getitem__(self, item=None):
        x, y = item[0], item[1]
        if x < 0 or y < 0:
            return np.zeros(len(self.pi))

        return self.pi[:, x, y]


