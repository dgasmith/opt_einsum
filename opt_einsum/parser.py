#!/usr/bin/env python
# coding: utf-8
"""
A functionally equivalent parser of the numpy.einsum input parser
"""

import numpy as np


einsum_symbols_base = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def is_valid_einsum_char(x):
    """Check if the character ``x`` is valid for numpy einsum.
    """
    return (x in einsum_symbols_base) or (x in ',->.')


def has_valid_einsum_chars_only(einsum_str):
    """Check if ``einsum_str`` contains only valid characters for numpy einsum.
    """
    return all(map(is_valid_einsum_char, einsum_str))


def get_symbol(i):
    """Get the symbol corresponding to int ``i`` - runs through the usual 52
    letters before resorting to unicode characters, starting at ``chr(192)``.

    Examples
    --------
    >>> get_symbol(2)
    'c'

    >>> oe.get_symbol(200)
    'Ŕ'

    >>> oe.get_symbol(20000)
    '京'
    """
    if i < 52:
        return einsum_symbols_base[i]
    return chr(i + 140)


def gen_unused_symbols(used, n):
    """Generate ``n`` symbols that are not already in ``used``.
    """
    i = cnt = 0
    while cnt < n:
        s = get_symbol(i)
        i += 1
        if s in used:
            continue
        yield s
        cnt += 1


def convert_to_valid_einsum_chars(einsum_str):
    """Convert the str ``einsum_str`` to contain only the alphabetic characters
    valid for numpy einsum.
    """
    # partition into valid and invalid sets
    valid, invalid = set(), set()
    for x in einsum_str:
        (valid if is_valid_einsum_char(x) else invalid).add(x)

    # get replacements for invalid chars that are not already used
    available = gen_unused_symbols(valid, len(invalid))

    # map invalid to available and replace in the inputs
    replacer = dict(zip(invalid, available))
    return "".join(replacer.get(x, x) for x in einsum_str)


def find_output_str(subscripts):
    """Find the output string for the inputs ``susbcripts``.
    """
    tmp_subscripts = subscripts.replace(",", "")
    return "".join(s for s in sorted(set(tmp_subscripts)) if tmp_subscripts.count(s) == 1)


def find_output_shape(inputs, shapes, output):
    """Find the output shape for given inputs, shapes and output string, taking
    into account broadcasting.
    """
    return tuple(
        max(shape[loc] for shape, loc in zip(shapes, [x.find(c) for x in inputs]) if loc >= 0)
        for c in output
    )


def possibly_convert_to_numpy(x):
    """Convert things without a 'shape' to ndarrays, but leave everything else.
    """
    if not hasattr(x, 'shape'):
        return np.asanyarray(x)
    else:
        return x


def parse_einsum_input(operands):
    """
    A reproduction of einsum c side einsum parsing in python.

    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the numpy contraction

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> parse_einsum_input(('...a,...a->...', a, b))
    ('za,xza', 'xz', [a, b])

    >>> parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('za,xza', 'xz', [a, b])
    """

    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = [possibly_convert_to_numpy(x) for x in operands[1:]]

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = [possibly_convert_to_numpy(x) for x in operand_list]
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "..."
                elif isinstance(s, int):
                    subscripts += get_symbol(s)
                else:
                    raise TypeError("For this input type lists must contain either int or Ellipsis")
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                elif isinstance(s, int):
                    subscripts += get_symbol(s)
                else:
                    raise TypeError("For this input type lists must contain either int or Ellipsis")

    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        ellipse_inds = "".join(gen_unused_symbols(used, max(len(x.shape) for x in operands)))
        longest = 0

        # Do we have an output to account for?
        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(len(operands[num].shape), 1) - (len(sub) - 3)

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    split_subscripts[num] = sub.replace('...', ellipse_inds[-ellipse_count:])

        subscripts = ",".join(split_subscripts)

        # Figure out output ellipses
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = find_output_str(subscripts)
            normal_inds = ''.join(sorted(set(output_subscript) - set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts, output_subscript = subscripts, find_output_str(subscripts)

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError("Output character %s did not appear in the input" % char)

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the " "number of operands.")

    return input_subscripts, output_subscript, operands
