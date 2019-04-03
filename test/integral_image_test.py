import unittest
import numpy as np
from integral_image import IntegralImage, \
    TwoRectangleVerticalFeature, TwoRectangleHorizontalFeature, \
    ThreeRectangleVerticalFeature, ThreeRectangleHorizontalFeature, \
    ThreeExtendedRectangleVerticalFeature, ThreeExtendedRectangleHorizontalFeature, \
    FourRectangleFeature, CenterSurroundedRectangleFeature


class IntegralImageTest(unittest.TestCase):

    def setUp(self):
        '''
        0---+---+---+---+      0---+---+---+---+
        | 1 | 1 | 1 | 1 |      | 1 | 2 | 3 | 4 |
        +---1---+---+---+      +---1---+---+---+
        | 1 | 1 | 1 | 1 |      | 2 | 4 | 6 | 8 |
        +---+---2---+---+  =>  +---+---2---+---+
        | 1 | 1 | 1 | 1 |      | 3 | 6 | 9 | 12|
        +---+---+---3---+      +---+---+---3---+
        | 1 | 1 | 1 | 1 |      | 4 | 8 | 12| 16|
        +---+---+---+---4      +---+---+---+---4
        '''

        self.ii = IntegralImage(np.ones((4, 4)))

    def test_structure(self):
        self.assertEqual(0.0, self.ii[-1, -1])
        self.assertEqual(0.0, self.ii[-1, 0])
        self.assertEqual(0.0, self.ii[0, -1])

        self.assertEqual(1.0, self.ii[0, 0])
        self.assertEqual(2.0, self.ii[0, 1])
        self.assertEqual(3.0, self.ii[0, 2])
        self.assertEqual(4.0, self.ii[0, 3])

        self.assertEqual(2.0, self.ii[1, 0])
        self.assertEqual(4.0, self.ii[1, 1])
        self.assertEqual(6.0, self.ii[1, 2])
        self.assertEqual(8.0, self.ii[1, 3])

        self.assertEqual(3.0, self.ii[2, 0])
        self.assertEqual(6.0, self.ii[2, 1])
        self.assertEqual(9.0, self.ii[2, 2])
        self.assertEqual(12.0, self.ii[2, 3])

        self.assertEqual(4.0, self.ii[3, 0])
        self.assertEqual(8.0, self.ii[3, 1])
        self.assertEqual(12.0, self.ii[3, 2])
        self.assertEqual(16.0, self.ii[3, 3])

    def test_sum_area_4(self):
        s = self.ii.sum_rectangle(x=1, y=1, w=1, h=1)
        self.assertEqual(1.0, s)

    def test_sum_area_4_6_6_9(self):
        s = self.ii.sum_rectangle(x=1, y=1, w=2, h=2)
        self.assertEqual(4.0, s)

    def test_sum_area_all(self):
        s = self.ii.sum_rectangle(x=0, y=0, w=4, h=4)
        self.assertEqual(16.0, s)


class IntegralImageTest2(unittest.TestCase):

    def setUp(self):
        '''
        0---+---+---+---+      0---+---+---+---+
        | 1 | 2 | 3 | 4 |      | 1 | 3 | 6 | 10|
        +---1---+---+---+      +---1---+---+---+
        | 5 | 6 | 7 | 8 |      | 6 | 14| 24| 36|
        +---+---2---+---+  =>  +---+---2---+---+
        | 9 | 10| 11| 12|      | 15| 33| 54| 78|
        +---+---+---3---+      +---+---+---3---+
        | 13| 14| 15| 16|      | 28| 60| 96|136|
        +---+---+---+---4      +---+---+---+---4
        '''

        self.ii = IntegralImage(np.array([x + 1 for x in range(16)],
                                         dtype=np.float64).reshape((4, 4)))

    def test_structure(self):
        self.assertEqual(0.0, self.ii[-1, -1])
        self.assertEqual(0.0, self.ii[0, -1])
        self.assertEqual(0.0, self.ii[-1, 0])

        self.assertEqual(1.0, self.ii[0, 0])
        self.assertEqual(3.0, self.ii[0, 1])
        self.assertEqual(6.0, self.ii[0, 2])
        self.assertEqual(10.0, self.ii[0, 3])

        self.assertEqual(6.0, self.ii[1, 0])
        self.assertEqual(14.0, self.ii[1, 1])
        self.assertEqual(24.0, self.ii[1, 2])
        self.assertEqual(36.0, self.ii[1, 3])

        self.assertEqual(15.0, self.ii[2, 0])
        self.assertEqual(33.0, self.ii[2, 1])
        self.assertEqual(54.0, self.ii[2, 2])
        self.assertEqual(78.0, self.ii[2, 3])

        self.assertEqual(28.0, self.ii[3, 0])
        self.assertEqual(60.0, self.ii[3, 1])
        self.assertEqual(96.0, self.ii[3, 2])
        self.assertEqual(136.0, self.ii[3, 3])

    def test_sum_area_6(self):
        s = self.ii.sum_rectangle(x=1, y=1, w=1, h=1)
        self.assertEqual(s, 6.0)

    def test_sum_area_6_7_10_11(self):
        s = self.ii.sum_rectangle(x=1, y=1, w=2, h=2)
        self.assertEqual(s, 34.0)

    def test_sum_area_all(self):
        s = self.ii.sum_rectangle(x=0, y=0, w=4, h=4)
        self.assertEqual(s, 136.0)


class FeatureTest(unittest.TestCase):

    def setUp(self):
        '''
        0---+---+---+---+      0---+---+---+---+
        | 1 | 2 | 3 | 4 |      | 1 | 3 | 6 | 10|
        +---1---+---+---+      +---1---+---+---+
        | 5 | 6 | 7 | 8 |      | 6 | 14| 24| 36|
        +---+---2---+---+  =>  +---+---2---+---+
        | 9 | 10| 11| 12|      | 15| 33| 54| 78|
        +---+---+---3---+      +---+---+---3---+
        | 13| 14| 15| 16|      | 28| 60| 96|136|
        +---+---+---+---4      +---+---+---+---4
        '''

        self.ii = IntegralImage(np.array([x + 1 for x in range(16)],
                                         dtype=np.float64).reshape((4, 4)))

    def test_TwoRectangleVerticalFeature(self):
        '''
        0---+---+---+---+
        |   |   |   |   |
        +---1---+---+---+
        |   | o | x |   |
        +---+---2---+---+
        |   | o | x |   |
        +---+---+---3---+
        |   | o | x |   |
        +---+---+---+---4

        (7+11+15) - (6+10+14) = 3
        '''

        feature = TwoRectangleVerticalFeature(x=1, y=1, w=2, h=3)
        s = feature.area(self.ii)
        self.assertEqual(3.0, s)

    def test_TwoRectangleHorizontalFeature(self):
        '''
        0---+---+---+---+
        |   |   |   |   |
        +---1---+---+---+
        |   | x | x | x |
        +---+---2---+---+
        |   | o | o | o |
        +---+---+---3---+
        |   |   |   |   |
        +---+---+---+---4

        (6+7+8) - (10+11+12) = -12
        '''

        feature = TwoRectangleHorizontalFeature(x=1, y=1, w=3, h=2)
        s = feature.area(self.ii)
        self.assertEqual(-12.0, s)

    def test_ThreeRectangleVerticalFeature(self):
        '''
        0---+---+---+---+
        |   |   |   |   |
        +---1---+---+---+
        |   | o | x | O |
        +---+---2---+---+
        |   | o | x | O |
        +---+---+---3---+
        |   |   |   |   |
        +---+---+---+---4

        (7+11) - (6+10) - (8+12) = -18
        '''

        feature = ThreeRectangleVerticalFeature(x=1, y=1, w=3, h=2)
        s = feature.area(self.ii)
        self.assertEqual(-18.0, s)

    def test_ThreeRectangleHorizontalFeature(self):
        '''
        0---+---+---+---+
        |   |   |   |   |
        +---1---+---+---+
        |   | o | o |   |
        +---+---2---+---+
        |   | x | x |   |
        +---+---+---3---+
        |   | O | O |   |
        +---+---+---+---4

        (10+11) - (6+7) - (14+15) = -21
        '''

        feature = ThreeRectangleHorizontalFeature(x=1, y=1, w=2, h=3)
        s = feature.area(self.ii)
        self.assertEqual(-21.0, s)

    def test_FourRectangleFeature(self):
        '''
        0---+---+---+---+
        |   |   |   |   |
        +---1---+---+---+
        |   | o | x |   |
        +---+---2---+---+
        |   | X | O |   |
        +---+---+---3---+
        |   |   |   |   |
        +---+---+---+---4

        (7+10) - (6+11) = 0.0
        '''

        feature = FourRectangleFeature(x=1, y=1, w=2, h=2)
        s = feature.area(self.ii)
        self.assertEqual(0.0, s)

    def test_ThreeExtendedRectangleVerticalFeature(self):
        '''
        0---+---+---+---+
        |   |   |   |   |
        +---1---+---+---+
        | o | x | x | O |
        +---+---2---+---+
        |   |   |   |   |
        +---+---+---3---+
        |   |   |   |   |
        +---+---+---+---4

        (6+7) - (5+8) = 0
        '''

        feature = ThreeExtendedRectangleVerticalFeature(x=0, y=1, w=4, h=1)
        s = feature.area(self.ii)
        self.assertEqual(0.0, s)

    def test_ThreeExtendedRectangleHorizontalFeature(self):
        '''
        0---+---+---+---+
        |   |   | o |   |
        +---1---+---+---+
        |   |   | x |   |
        +---+---2---+---+
        |   |   | x |   |
        +---+---+---3---+
        |   |   | O |   |
        +---+---+---+---4


        (7+11) - (3+15) = 0
        '''

        feature = ThreeExtendedRectangleHorizontalFeature(x=2, y=0, w=1, h=4)
        s = feature.area(self.ii)
        self.assertEqual(0.0, s)

    def test_CenterSurroundedRectangleFeature(self):
        '''
        0---+---+---+---+
        |   |   |   |   |
        +---1---+---+---+
        |   | a | b | c |
        +---+---2---+---+
        |   | d | x | e |
        +---+---+---3---+
        |   | f | g | h |
        +---+---+---+---4

        11 - (6+7+8+10+12+14+15+16) = -77
        '''

        feature = CenterSurroundedRectangleFeature(x=1, y=1, w=3, h=3)
        s = feature.area(self.ii)
        self.assertEqual(-77.0, s)
