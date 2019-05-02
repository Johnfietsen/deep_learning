# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...

        # for use in forward pass
        self._device = device

        # stacked LSTMs
        self._lstm = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers,
                             dropout=0.5)

        # output layer
        self._linear = nn.Linear(lstm_num_hidden, vocabulary_size)


    def forward(self, x):
        # Implementation here...

        out, (h, c) = self._lstm(x)

        return self._linear(out).to(self._device), h, c


    def generate_character(self, x, h, c):

        out, (h, c) = self._lstm(x, (h, c))

        return self._linear(out).to(self._device), h, c
