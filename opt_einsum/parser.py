"""
A functionally equivalent parser of the numpy.einsum input parser
"""

import sys

import numpy as np

einsum_symbols_base = 'abcdefghijklmnopqrstuvwxyz'
einsum_symbols = einsum_symbols_base + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# boost the number of symbols using unicode if python3
if sys.version_info[0] >= 3:
    einsum_symbols += ''.join(map(chr, range(193, 688)))
    einsum_symbols += ''.join(map(chr, range(913, 1367)))

einsum_symbols_set = set(einsum_symbols)


def is_valid_einsum_char(x):
    """Check if the character ``x`` is valid for numpy einsum.
    """
    return (x in einsum_symbols_base) or (x in ',->.')


def has_valid_einsum_chars_only(einsum_str):
    """Check if ``einsum_str`` contains only valid characters for numpy einsum.
    """
    return all(map(is_valid_einsum_char, einsum_str))


def convert_to_valid_einsum_chars(einsum_str):
    """Convert the str ``einsum_str`` to contain only the alphabetic characters
    valid for numpy einsum.
    """
    # partition into valid and invalid sets
    valid, invalid = set(), set()
    for x in einsum_str:
        (valid if is_valid_einsum_char(x) else invalid).add(x)

    # get replacements for invalid chars that are not already used
    available = (x for x in einsum_symbols if x not in valid)

    # map invalid to available and replace in the inputs
    replacer = dict(zip(invalid, available))
    return "".join(replacer.get(x, x) for x in einsum_str)


def find_output_str(subscripts):
    """Find the output string for the inputs ``susbcripts``.
    """
    tmp_subscripts = subscripts.replace(",", "")
    output_subscript = ""
    for s in sorted(set(tmp_subscripts)):
        if s not in einsum_symbols_set:
            raise ValueError("Character %s is not a valid symbol." % s)
        if tmp_subscripts.count(s) == 1:
            output_subscript += s
    return output_subscript


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
    >>> _parse_einsum_input(('...a,...a->...', a, b))
    ('za,xza', 'xz', [a, b])

    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('za,xza', 'xz', [a, b])
    """

    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = [possibly_convert_to_numpy(x) for x in operands[1:]]

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols_set:
                raise ValueError("Character %s is not a valid symbol." % s)

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
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists must contain " "either int or Ellipsis")
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists must contain " "either int or Ellipsis")
    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
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
