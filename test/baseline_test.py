#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import torch
import numpy as np
import random

import models

class BaselineTest(unittest.TestCase):
    def setUp(self):
        # setup the device
        self.device = torch.device("cpu")
        random.seed(41)
        np.random.seed(41)
        torch.manual_seed(41)

    def testForward(self):
        """ This test hardcodes values from the working code.
        """
        word_vectors = torch.tensor([[1,1], [2, 2], [3,3], [4,4]], dtype=torch.float32)
        model = models.BiDAF(word_vectors,1,0)
        input = torch.tensor([[2,2],[3,3]], dtype=torch.long)
        output = model(input, qw_idxs=torch.tensor([[2,2],[3,3]], dtype=torch.long))
        torch.testing.assert_allclose(output[0], [[-0.6976, -0.6887],
            [-0.6979, -0.6884]], 0.1, 0.1)
        torch.testing.assert_allclose(output[1], [[-0.6976, -0.6887],
            [-0.6979, -0.6884]], 0.1, 0.1)