# Python 2/3 compatability shim

try:
    # Python 3
    chr(256)
    get_chr = chr
    strings = str
except ValueError:
    # Python 2

    def get_chr(i):
        return chr(i) if i < 128 else unichr(i)

    strings = (str, type(get_chr(300)))
