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

        # store for use in forward pass
        self._seq_length = seq_length
        self._num_hidden = num_hidden
        self._batch_size = batch_size

        # recurrent part
        self._Whx = nn.Parameter(torch.zeros(input_dim, num_hidden))
        self._Whh = nn.Parameter(torch.zeros(num_hidden, num_hidden))
        self._bh = nn.Parameter(torch.zeros(batch_size, num_hidden))

        # output
        self._Wph = nn.Parameter(torch.zeros(num_hidden, num_classes))
        self._bp = nn.Parameter(torch.zeros(batch_size, num_classes))

        # initialize randomly
        nn.init.kaiming_normal_(self._Whx)
        nn.init.kaiming_normal_(self._Whh)
        nn.init.kaiming_normal_(self._Wph)

    def forward(self, x):

        # initialize hidden state
        h = torch.zeros(self._batch_size, self._num_hidden)

        # loop through sequence
        for t in range(self._seq_length):
            h = torch.tanh(x[:, t, None] @ self._Whx + h @ self._Whh + self._bh)

        # calculate p
        return h @ self._Wph + self._bp
