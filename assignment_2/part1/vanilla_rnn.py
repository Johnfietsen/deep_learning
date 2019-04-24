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

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, \
                 batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        # save values for later use
        self._seq_length = seq_length
        self._batch_size = batch_size

        # input-to-hidden
        self._Whx = nn.Parameter(torch.tensor((num_hidden, input_dim))\
                                 .random_())

        # hidden-to-hidden
        self._Whh = nn.Parameter(torch.tensor((num_hidden, num_hidden))\
                                 .random_())

        # bias
        self._bh = nn.Parameter(torch.tensor((num_hidden, 1)).random_())

        # hidden-to-output
        self._Wph = nn.Parameters(torch.tensor((num_classes, num_hidden))\
                                  .random_())

        # bias
        self._bp = nn.Parameter(torch.tensor((num_classes, 1)).random_())

        self._h = [torch.tensor((num_hidden, 1)).random_()]
        self._p = [torch.tensor((num_classes, 1)).random_()]

    def forward(self, x):

        for i in range(1, self._seq_length + 1):
            self._h.append(nn.tanh(self._Whx @ x \
                           + self_Whh @ self._h[i - 1] + self._bh))
            self._p.append(self._Wph @ self._h[i] + self._bp)
