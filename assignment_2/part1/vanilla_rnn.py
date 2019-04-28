################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import copy as cp

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, \
                 batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        # save values for later use
        self._seq_length = seq_length
        self._input_dim = input_dim
        self._num_hidden = num_hidden
        self._batch_size = batch_size

        a = 1 / num_hidden
        b = 1 / num_classes

        # input-to-hidden
        self._Whx = nn.Parameter(a * torch.randn((input_dim, num_hidden)))

        # hidden-to-hidden
        self._Whh = nn.Parameter(a * torch.randn((num_hidden, num_hidden)))

        # bias
        self._bh = nn.Parameter(a * torch.randn((num_hidden, 1)))

        # hidden-to-output
        self._Wph = nn.Parameter(a * torch.randn((num_classes, num_hidden)))

        # bias
        self._bp = nn.Parameter(b * torch.randn((num_classes, 1)))


    def forward(self, x):

        # initialize hidden state
        h = torch.zeros(self._num_hidden, self._batch_size)

        # loop through sequence
        for i in range(self._seq_length):

            # calculate h
            h = torch.tanh(self._Whx @ x[:, i].view(-1, self._input_dim) +
                           self._Whh @ h + self._bh)

        # return p
        return torch.transpose(self._Wph @ h + self._bp, 0, 1)
