import unittest
import numpy as np
from integral_image import IntegralImage, TiltedIntegralImage, \
    TwoRectangleVerticalFeature, TwoRectangleHorizontalFeature, \
    ThreeRectangleVerticalFeature, ThreeRectangleHorizontalFeature, \
    ThreeExtendedRectangleVerticalFeature, ThreeExtendedRectangleHorizontalFeature, \
    FourRectangleFeature, CenterSurroundedRectangleFeature


class FeatureSelectionTest(unittest.TestCase):

    def setUp(self):
        self.shape = (24, 24)
        self.ii = IntegralImage((np.ones(self.shape)))
        self.tii = TiltedIntegralImage((np.ones(self.shape)))

    def test_select_TwoRectangleVerticalFeature(self):
        features = self.ii.generate_features(self.shape, feature_class=TwoRectangleVerticalFeature)
        self.assertEqual(43200, len(features))

    def test_select_TwoRectangleHorizontalFeature(self):
        features = self.ii.generate_features(self.shape, feature_class=TwoRectangleHorizontalFeature)
        self.assertEqual(43200, len(features))

    def test_select_ThreeRectangleVerticalFeature(self):
        features = self.ii.generate_features(self.shape, ThreeRectangleVerticalFeature)
        self.assertEqual(27600, len(features))

    def test_select_ThreeRectangleHorizontalFeature(self):
        features = self.ii.generate_features(self.shape, ThreeRectangleHorizontalFeature)
        self.assertEqual(27600, len(features))

    def test_select_ThreeExtendedRectangleVerticalFeature(self):
        features = self.ii.generate_features(self.shape, ThreeExtendedRectangleVerticalFeature)
        self.assertEqual(19800, len(features))

    def test_select_ThreeExtendedRectangleHorizontalFeature(self):
        features = self.ii.generate_features(self.shape, ThreeExtendedRectangleHorizontalFeature)
        self.assertEqual(19800, len(features))

    def test_select_FourRectangleFeature(self):
        features = self.ii.generate_features(self.shape, FourRectangleFeature)
        self.assertEqual(20736, len(features))

    def test_select_CenterSurroundedRectangleFeature(self):
        features = self.ii.generate_features(self.shape, CenterSurroundedRectangleFeature)
        self.assertEqual(8464, len(features))
