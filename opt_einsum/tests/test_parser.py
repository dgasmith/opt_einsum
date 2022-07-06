"""
Directly tests various parser utility functions.
"""

from opt_einsum.parser import get_symbol


def test_get_symbol():
    assert get_symbol(2) == "c"
    assert get_symbol(200000) == "\U00031540"
    # Ensure we skip surrogates '[\uD800-\uDFFF]'
    assert get_symbol(55295) == "\ud88b"
    assert get_symbol(55296) == "\ue000"
    assert get_symbol(57343) == "\ue7ff"
