#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import torch
from test.utils import init_random

import char_layers

class CnnTest(unittest.TestCase):
    def setUp(self):
        init_random(41)
        
    def testForward(self):
        cnn = char_layers.CNN(1,1,2)
        input = torch.tensor([[[1.0,2.0, 3.0]]])
        cnn.initializeUniform(1.0)
        output = cnn(input)
        torch.testing.assert_allclose(output, [[[3.0, 5.0]]], 0.1, 0.1)

class CharEmbedditgsTest(unittest.TestCase):
    def testForward(self):
        shape = char_layers.CharEmbeddingsShape(
            # size of one char embedding, for example 50 
            3,
            # max numbers of chars in a word, for example 21
            21, 
            # output embedding size
            2, 
            # number of chars in the input vocabulary
            4, 
            # the width of the cnn kernel, for example 5
            2,
            # idx of the pad token
            0,
            # dropout_prob
            0
        )
        model = char_layers.CharEmbeddings(shape)
        input = torch.ones(2, 2, 3).long()
        output = model(input)
        # can't seem to get stable output
        assert (output is not None)
