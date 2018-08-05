"""
Required functions for optimized contractions of numpy arrays using pytorch.
"""

from __future__ import absolute_import
import numpy as np

from ..parser import einsum_symbols_base


_TORCH_DEVICE = None


def _get_torch_and_device():
    global _TORCH_DEVICE

    if _TORCH_DEVICE is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _TORCH_DEVICE = torch, device

    return _TORCH_DEVICE


def transpose(a, axes):
    """Normal torch transpose is only valid for 2D matrices.
    """
    return a.permute(*axes)


def einsum(equation, *operands):
    """Variadic version of torch.einsum to match numpy api.
    """
    torch, _ = _get_torch_and_device()
    return torch.einsum(equation, operands)


def tensordot(x, y, axes=2):
    """Simple translation of tensordot syntax to einsum.
    """
    # XXX: tensordot should be directly implemented in torch soon
    torch, _ = _get_torch_and_device()

    xnd = x.ndimension()
    ynd = y.ndimension()

    # convert int argument to (list[int], list[int])
    if isinstance(axes, int):
        axes = range(xnd - axes, xnd), range(axes)

    # convert (int, int) to (list[int], list[int])
    if isinstance(axes[0], int):
        axes = (axes[0],), axes[1]
    if isinstance(axes[1], int):
        axes = axes[0], (axes[1],)

    # initialize empty indices
    x_ix = [None] * xnd
    y_ix = [None] * ynd
    out_ix = []

    # fill in repeated indices
    available_ix = iter(einsum_symbols_base)
    for ax1, ax2 in zip(*axes):
        repeat = next(available_ix)
        x_ix[ax1] = repeat
        y_ix[ax2] = repeat

    # fill in the rest, and maintain output order
    for i in range(xnd):
        if x_ix[i] is None:
            leave = next(available_ix)
            x_ix[i] = leave
            out_ix.append(leave)
    for i in range(ynd):
        if y_ix[i] is None:
            leave = next(available_ix)
            y_ix[i] = leave
            out_ix.append(leave)

    # form full string and contract!
    einsum_str = "{},{}->{}".format(*map("".join, (x_ix, y_ix, out_ix)))
    return torch.einsum(einsum_str, (x, y))


def to_torch(array):
    torch, device = _get_torch_and_device()

    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(device)

    return array


def build_expression(_, expr):  # pragma: no cover
    """Build a torch function based on ``arrays`` and ``expr``.
    """

    def torch_contract(*arrays):
        torch_arrays = [to_torch(x) for x in arrays]
        torch_out = expr._contract(torch_arrays, backend='torch')

        if torch_out.device.type == 'cpu':
            return torch_out.numpy()

        return torch_out.cpu().numpy()

    return torch_contract


def evaluate_constants(const_arrays, expr):
    """Convert constant arguments to torch, and perform any possible constant
    contractions.
    """
    const_arrays = [to_torch(x) for x in const_arrays]
    return expr(*const_arrays, backend='torch', evaluate_constants=True)
