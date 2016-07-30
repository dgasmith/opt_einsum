from __future__ import division, absolute_import, print_function

import numpy as np

chars = 'abcdefghijklm'
sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5])
dimension_dict = {c : s for c, s in zip(chars, sizes)}

def build_views(string):

    views = []
    terms = string.split('->')[0].split(',')
    for term in terms:
        dims = [dimension_dict[x] for x in term]
        views.append(np.random.rand(*dims))
    return views
