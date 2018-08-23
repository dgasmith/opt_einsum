# Python 2/3 compatability shim

try:
    # Python 2
    get_chr = unichr
    strings = (str, type(get_chr(300)))
except NameError:
    # Python 3
    get_chr = chr
    strings = str
