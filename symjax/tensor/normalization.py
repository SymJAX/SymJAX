from . import linalg
from .ops_special import reshape_weight_to_matrix, stop_gradient

# Normalization


def normalize(input, p=2, dim=1, eps=1e-12):
    denom = linalg.norm(input, p, dim, keepdim=True)
    return denom / T.maximum(denom, eps)


def spectral_normalize(weight, axis):
    """Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
    """
    weight_mat = reshape_weight_to_matrix(weight, axis)
    u, v = linalg.singular_vectors_power_iteration(weight_mat, axis)
    sigma = stop_gradient(u).dot(weight_mat.dot(stop_gradient(v)))
    return weight / sigma
