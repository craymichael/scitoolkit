# =====================================================================
# metrics_test.py - A scitoolkit file
# Copyright (C) 2018  Zach Carmichael
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =====================================================================
from scitoolkit.util.py23 import *  # py2/3 compatibility

from scitoolkit.infrastructure import test
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
