#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2019 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

import warnings
import numpy as np
from typing import Optional

from ..pipeline import Pipeline
from ..parameter import Uniform
from pyannote.core.utils.distance import cdist
from pyannote.core.utils.distance import dist_range
from pyannote.core.utils.distance import l2_normalize


class ClosestAssignment(Pipeline):
    """Assign each sample to the closest target

    Parameters
    ----------
    metric : `str`, optional
        Distance metric. Defaults to 'cosine'
    normalize : `bool`, optional
        L2 normalize vectors before clustering.

    Hyper-parameters
    ----------------
    threshold : `float`
        Do not assign if distance greater than `threshold`.
    """

    def __init__(self, metric: Optional[str] = 'cosine',
                       normalize: Optional[bool] = False):

        super().__init__()
        self.metric = metric
        self.normalize = normalize

        min_dist, max_dist = dist_range(metric=self.metric,
                                        normalize=self.normalize)
        if not np.isfinite(max_dist):
            # this is arbitray and might lead to suboptimal results
            max_dist = 1e6
            msg = (f'bounding distance threshold to {max_dist:g}: '
                   f'this might lead to suboptimal results.')
            warnings.warn(msg)
        self.threshold = Uniform(min_dist, max_dist)

    def __call__(self, X_target, X):
        """Assign each sample to its closest class (if close enough)

        Parameters
        ----------
        X_target : `np.ndarray`
            (n_targets, n_dimensions) target embeddings
        X : `np.ndarray`
            (n_samples, n_dimensions) sample embeddings

        Returns
        -------
        assignments : `np.ndarray`
            (n_samples, ) sample assignments
        """

        if self.normalize:
            X_target = l2_normalize(X_target)
            X = l2_normalize(X)

        distance = cdist(X_target, X, metric=self.metric)
        targets = np.argmin(distance, axis=0)

        for i, k in enumerate(targets):
            if distance[k, i] > self.threshold:
                # do not assign
                targets[i] = -i

        return targets
