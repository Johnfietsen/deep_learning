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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, \
                 batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...

        # store for use in forward pass
        self._seq_length = seq_length
        self._num_hidden = num_hidden
        self._batch_size = batch_size
        self._device = device

        # input modulation gate
        self._Wgx = nn.Parameter(torch.zeros(input_dim, num_hidden))
        self._Wgh = nn.Parameter(torch.zeros(num_hidden, num_hidden))
        self._bg = nn.Parameter(torch.zeros(batch_size, num_hidden))

        # input gate
        self._Wix = nn.Parameter(torch.zeros(input_dim, num_hidden))
        self._Wih = nn.Parameter(torch.zeros(num_hidden, num_hidden))
        self._bi = nn.Parameter(torch.zeros(batch_size, num_hidden))

        # forget gate
        self._Wfx = nn.Parameter(torch.zeros(input_dim, num_hidden))
        self._Wfh = nn.Parameter(torch.zeros(num_hidden, num_hidden))
        self._bf = nn.Parameter(torch.zeros(batch_size, num_hidden))

        # output gate
        self._Wox = nn.Parameter(torch.zeros(input_dim, num_hidden))
        self._Woh = nn.Parameter(torch.zeros(num_hidden, num_hidden))
        self._bo = nn.Parameter(torch.zeros(batch_size, num_hidden))

        # output
        self._Wph = nn.Parameter(torch.zeros(num_hidden, num_classes))
        self._bp = nn.Parameter(torch.zeros(batch_size, num_classes))

        # initialize randomly
        nn.init.kaiming_normal_(self._Wgx)
        nn.init.kaiming_normal_(self._Wgh)
        nn.init.kaiming_normal_(self._Wix)
        nn.init.kaiming_normal_(self._Wih)
        nn.init.kaiming_normal_(self._Wfx)
        nn.init.kaiming_normal_(self._Wfh)
        nn.init.kaiming_normal_(self._Wox)
        nn.init.kaiming_normal_(self._Woh)
        nn.init.kaiming_normal_(self._Wph)

        self.to(device)

    def forward(self, x):

        # initialize hidden state
        h = torch.zeros(self._batch_size, self._num_hidden).to(self._device)
        c = torch.zeros(self._batch_size, self._num_hidden).to(self._device)

        # loop through sequence
        for t in range(self._seq_length):

            # candidate gate
            g = torch.tanh(x[:, t, None] @ self._Wgx + h @ self._Wgh + self._bg)
            g = g.to(self._device)

            # input gate
            i = torch.sigmoid(x[:, t, None] @ self._Wix + h @ self._Wih \
                              + self._bi)
            i = i.to(self._device)

            # forget gate
            f = torch.sigmoid(x[:, t, None] @ self._Wfx + h @ self._Wfh \
                              + self._bf)
            f = f.to(self._device)

            # output gate
            o = torch.sigmoid(x[:, t, None] @ self._Wox + h @ self._Woh \
                              + self._bo)
            o = o.to(self._device)

            # hidden state
            c = g * i + c * f
            c = c.to(self._device)
            h = torch.tanh(c) * o
            h = h.to(self._device)


        # calculate p
        return (h @ self._Wph + self._bp).to(self._device)
