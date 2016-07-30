from __future__ import division, absolute_import, print_function

import numpy as np

chars = 'abcdefghijklm'
sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5])
default_dim_dict = {c : s for c, s in zip(chars, sizes)}

def build_views(string, dimension_dict=default_dim_dict):
    """
    Builds random numpy arrays for testing.

    Parameters
    ----------
    tensor_list : list of str
        List of tensor strings to build
    dimension_dictionary : dictionary
        Dictionary of index sizes

    Returns
    -------
    ret : list of np.ndarry's
        The resulting views.

    Examples
    --------
    >>> view = build_views(['abbc'], {'a': 2, 'b':3, 'c':5})
    >>> view[0].shape
    (2, 3, 3, 5)

    """

    views = []
    terms = string.split('->')[0].split(',')
    for term in terms:
        dims = [dimension_dict[x] for x in term]
        views.append(np.random.rand(*dims))
    return views
