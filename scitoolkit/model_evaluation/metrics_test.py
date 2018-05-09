from scitoolkit.py23 import *  # py2/3 compatibility

from scitoolkit.platform import test
from scitoolkit.model_evaluation import metrics
import numpy as np


class MeanPerClassAccuracyTestCase(test.TestCase):

    def test_baseline(self):
        y_true = np.asarray([0, 1, 2, 3, 4, 5])
        y_pred = np.asarray([0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(
            metrics.mean_per_class_accuracy(y_true, y_pred,
                                            n_classes=6,
                                            labels=[0, 1, 2, 3, 4, 5]),
            1.)

    def test_n_class_inference(self):
        y_true = np.asarray([0, 1, 2, 3, 4, 5])
        y_pred = np.asarray([0, 1, 2, 0, 0, 0])
        self.assertAlmostEqual(
            metrics.mean_per_class_accuracy(y_true, y_pred,
                                            n_classes=None, labels=None),
            0.5)

    def test_n_class_truncated(self):
        y_true = np.asarray([0, 1, 2, 3, 4, 5])
        y_pred = np.asarray([0, 1, 2, 5, 3, 4])
        self.assertAlmostEqual(
            metrics.mean_per_class_accuracy(y_true, y_pred,
                                            n_classes=3, labels=None),
            1.)

    def test_n_class_inference_array_origin(self):
        y_true = np.asarray([0, 1, 2, 0, 0, 0])
        y_pred = np.asarray([0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(
            metrics.mean_per_class_accuracy(y_true, y_pred,
                                            n_classes=None, labels=None),
            0.75)

    def test_n_class_no_ground_truth(self):
        y_true = np.asarray([0, 1, 2, 0, 0, 0])
        y_pred = np.asarray([0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(
            metrics.mean_per_class_accuracy(y_true, y_pred,
                                            n_classes=6, labels=None),
            0.75)

    def test_arg_check(self):
        y_true = np.asarray([0, 1, 2, 0, 0, 0])
        y_pred = np.asarray([0, 1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            metrics.mean_per_class_accuracy(y_true, y_pred,
                                            n_classes=2, labels=[1, 2, 3])
