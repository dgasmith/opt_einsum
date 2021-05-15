# Input Format

The `opt_einsum` package was originally designed as a drop-in replacement for the `np.einsum`
function and supports all input formats that `np.einsum` supports. There are
two styles of input accepted, a basic introduction to which can be found in the
documentation for `numpy.einsum`. In addition to this, `opt_einsum`
extends the allowed index labels to unicode or arbitrary hashable, comparable
objects in order to handle large contractions with many indices.


## 'Equation' Input

As with `numpy.einsum`, here you specify an equation as a string,
followed by the array arguments:

```python
import opt_einsum as oe
eq = 'ijk,jkl->li'
x, y = np.random.rand(2, 3, 4), np.random.rand(3, 4, 5)
z = oe.contract(eq, x, y)
z.shape
#> (5, 2)
```

However, in addition to the standard alphabet, `opt_einsum` also supports
unicode characters:

```python
eq = "αβγ,βγδ->δα"
oe.contract(eq, x, y).shape
#> (5, 2)
```

This enables access to thousands of possible index labels. One way to access
these programmatically is through the function [`get_symbols`](../api_reference.md#opt_einsumget_symbol):

```python
oe.get_symbol(805)
#> 'α'
```

which maps an `int` to a unicode characater. Note that as with
`numpy.einsum` if the output is not specified with `->` it will default
to the sorted order of all indices appearing once:

```python
eq = "αβγ,βγδ"  # "->αδ" is implicit
oe.contract(eq, x, y).shape
#> (2, 5)
```


## 'Interleaved' Input

The other input format is to 'interleave' the array arguments with their index
labels ('subscripts') in pairs, optionally specifying the output indices as a
final argument. As with `numpy.einsum`, integers are allowed as these
index labels:

```python
oe.contract(x, [1, 2, 3], y, [2, 3, 4], [4, 1]).shape
#> (5, 2)
```

with the default output order again specified by the sorted order of indices
appearing once. However, unlike `numpy.einsum`, in `opt_einsum` you can
also put *anything* hashable and comparable such as `str` in the subscript list.
A simple example of this syntax is:

```python
x, y, z = np.ones((1, 2)), np.ones((2, 2)), np.ones((2, 1))
oe.contract(x, ('left', 'bond1'), y, ('bond1', 'bond2'), z, ('bond2', 'right'), ('left', 'right'))
#> array([[4.]])
```

The subscripts need to be hashable so that `opt_einsum` can efficiently process them, and
they should also be comparable so as to allow a default sorted output. For example:

```python
x = np.array([[0, 1], [2, 0]])

# original matrix
oe.contract(x, (0, 1))
#> array([[0, 1],
#>        [2, 0]])

# the transpose
oe.contract(x, (1, 0))
#> array([[0, 2],
#>        [1, 0]])

# original matrix, consistent behavior
oe.contract(x, ('a', 'b'))
#> array([[0, 1],
#>        [2, 0]])

# the transpose, consistent behavior
>>> oe.contract(x, ('b', 'a'))
#> array([[0, 2],
#>        [1, 0]])

# relative sequence undefined, can't determine output
>>> oe.contract(x, (0, 'a'))
#> TypeError: For this input type lists must contain either Ellipsis
#> or hashable and comparable object (e.g. int, str)
```

