import numpy as np


# RBF Layer

class RBF(object):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centers: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        widths: the learnable scaling factors of shape (out_features).
            The values are initialised as ones.

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, basis_func):
        self.in_features = in_features
        self.out_features = out_features
        self.centers, self.widths = self.reset_parameters()
        self.basis_func = basis_func

    def reset_parameters(self):
        centers = np.random.randn(self.out_features, self.in_features)
        widths = np.ones([self.out_features, ])
        return centers, widths

    def eval_basis(self, input): # (B, x)
        num_batch = np.shape(input)[0]
        size = (num_batch, self.out_features, self.in_features)  # (B, y, x)
        x = np.repeat(input.reshape([-1, 1, self.in_features]), self.out_features, axis=1)   # (B, y, x)
        c = np.repeat(self.centers.reshape([-1, self.out_features, self.in_features]), num_batch, axis=0) # (B, y, x)
        distances = np.sum((x - c) ** 2, axis=2) ** (0.5) * self.widths.reshape(-1, self.out_features) # (B, y)
        return self.basis_func(distances)


# RBFs

def gaussian(alpha):
    phi = np.exp(-1 * alpha ** 2)
    return phi


def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """

    bases = {'gaussian': gaussian}
    return bases


