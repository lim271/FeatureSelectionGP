# -*- coding: utf-8 -*-

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#         Vincent Dubourg <vincent.dubourg@gmail.com>
#         (mostly translation, see implementation details)
# License: BSD 3 clause

"""
The :mod:`sklearn.gaussian_process` module implements Gaussian Process
based regression and classification.
"""

from .feature_selection_gpr import FeatureSelectionGPR
from . import kernels


__all__ = ['FeatureSelectionGPR', 'kernels']