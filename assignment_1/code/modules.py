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
        max_x = np.reshape(x.max(axis=1), (x.shape[0], 1))

        nume = np.exp(x - max_x)
        self._out = nume / np.reshape(np.sum(nume, axis=1), (x.shape[0], 1))
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
        print('self._out', self._out.shape)
        print('dout', dout.shape)

        tmp_tensor = np.array([[[]]])

        i = 0
        for point in self._out:
            tmp_tensor,append(self._out @ self._out.T)
            i += 1
        # zeros((self._out.shape[0], self._out.shape[1],\
        #                        self._out.shape[1]))
        print('tmp', tmp_tensor.shape)

        # print(self._out.shape) # (7, 52)
        # dx = dout.T @ (- self._out ** 2 + np.diag(self._out))
        # print(dx.shape)
        # print(self._out.shape)
        # dx = - self._out @ self._out.T
        # print(dx.shape)
        ########################
        # END OF YOUR CODE    #
        #######################


        # #crazy shit over here!
        #
        # #explicit dimensions
        # batch_size = self._out.shape[0] #mini batch size
        # dim = self._out.shape[1] #feature dimension
        #
        # #creates a tensor with self.out elements in the diagonal
        # diag_xN = np.zeros((batch_size, dim, dim))
        # ii = np.arange(dim)
        # diag_xN[:, ii, ii] = self._out
        #
        # #einstein sum convention to the rescue! :sunglasses:
        # #first we calculate the dx/d\tilde{x}
        # dxdx_t = diag_xN - np.einsum('ij, ik -> ijk', self._out, self._out)
        #
        #
        # dx = np.einsum('ij, ijk -> ik', dout, dxdx_t)

        print('dx', dx.shape)
        # print('dxdx_t', dxdx_t.shape)
        # print('diag_xN', diag_xN.shape)

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
        out = - np.sum(y * np.log(x))
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
        dx = - y / x
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx
