"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        TODO:
        Initialize weights self.params['weight'] using normal distribution with
        mean = 0 and std = 0.0001. Initialize biases self.params['bias'] with 0.

        Also, initialize gradients with zeros.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.params = {'weight': np.random.normal(0, 0.0001, \
                                                  (out_features, in_features)),
                       'bias': np.zeros((out_features, 1))}
        self.grads = {'weight': np.zeros((out_features, in_features)),
                      'bias': np.zeros((out_features, 1))}
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can
        be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self._x = x
        out = x @ self.params['weight'].T + self.params['bias'].T
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with
        respect to layer parameters in self.grads['weight'] and
        self.grads['bias'].
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight'] = dout.T @ self._x
        self.grads['bias'] = dout.T @ np.ones((dout.shape[0], 1))
        dx = dout @ self.params['weight']
        print('\n', self.params['weight'])
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx

class ReLUModule(object):
    """
    ReLU activation module.
    """
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can
        be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self._out = np.maximum(0, x)
        ########################
        # END OF YOUR CODE    #
        #######################

        return self._out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = dout * (self._out > 0)
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx

class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can
        be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        sz = x.shape[0]
        max_x = np.reshape(x.max(axis=1), (sz, 1))

        num = np.exp(x - max_x)
        self._out = num / np.reshape(np.sum(num, axis=1), (sz, 1))
        ########################
        # END OF YOUR CODE    #
        #######################

        return self._out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        diagonal = [np.diag(x) for x in self._out]

        derivative = diagonal - np.einsum('ij, ik -> ijk', self._out, self._out)

        dx = np.einsum('ij, ijk -> ik', dout, derivative)
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx

class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    def forward(self, x, y):
        """
        Forward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        # normalize?
        out = - np.sum(y * np.log(x + 1e-5)) / len(y)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        # normalize?
        dx = - y / (len(y) * (x + 1e-5))
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx
