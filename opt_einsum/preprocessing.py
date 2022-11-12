import operator
import functools

from .backends import get_func
from .parser import find_output_str


def multiply(*xs):
    """Functional version of ``x[0] * x[1] * ...``.
    """
    return functools.reduce(operator.mul, xs)


class Lazy:
    """A tiny lazy computation class."""

    __slots__ = ("inds", "f", "args")

    def __init__(self, inds, f=None, *args):
        self.inds = inds
        self.f = f
        self.args = args

    def compute(self):
        return self.f(
            *(
                arg.compute() if isinstance(arg, Lazy) else arg
                for arg in self.args
            )
        )

    def __repr__(self):
        return f"Lazy('{self.inds}', {self.f}, {self.args})"


def make_simplifier(*args, backend="numpy"):
    """Take a einsum equation / specfification and return a new minimial
    'processed' specfication and a function which will perform said processing.
    The simplifications are as follows:

        - All indices which only appear on a single input (and not the output)
          are summed over.
        - All indices which appear multiple times on the same term are traced.
        - All scalars are multiplied into the smallest other term
        - All terms with the same indices are multiplied (hamadard product
          / elementwise) into a single term.

    Parameters
    ----------
    args : str, or tuple
        The einsum equation. It can already be split into inputs and output,
        and the inputs can already be split into a list of terms.
    backend : str, optional
        The backend to use for any ``einsum`` operations (tracing and summing).

    Returns
    -------
    new_spec : str, or tuple
        The new specification, with all the simplifications applied, in the
        same format as ``args`` was supplied.
    simplifier : callable
        A function which takes a list of arrays and returns the result of
        applying the simplifications to the arrays, i.e. compatible with the
        new einsum specification also returned.

    Examples
    --------

        >>> eq = ',ab,abee,,cd,cd,dd->ac'
        >>> arrays = helpers.build_views(eq)
        >>> new_eq, simplifier = make_simplifier(eq)
        >>> new_eq
        'ab,cd,d->ac'

        >>> sarrays = simplifier(arrays)
        >>> oe.contract(new_eq, *sarrays)
        array([[0.65245661, 0.14684493, 0.42543411, 0.32350895],
               [0.38357005, 0.08632807, 0.25010672, 0.19018636]])
    """
    try:
        # input and output already parsed
        format = 0
        terms, output = args
        if isinstance(terms, str):
            # input not split into list yet
            format = 1
            terms = terms.split(",")
    except ValueError:
        # single equation form
        (eq,) = args
        try:
            # with output specified
            format = 2
            inputs, output = eq.split("->")
        except ValueError:
            # not output specified
            format = 3
            inputs = eq
            output = find_output_str(inputs)
        terms = inputs.split(",")

    # want to make everything lazy
    lazy_inputs = [Lazy(term, lambda x: x) for term in terms]
    lazy_temps = lazy_inputs.copy()

    # first work out which indices only appear on a single term
    appears_once = set(output)  # ... and not the output
    appears_many = set()
    uterms = []
    for term in terms:
        uterm = []
        for ind in term:
            if ind not in uterm:
                # also build the unique inds of each term
                uterm.append(ind)
            else:
                # already check for this term
                continue
            if ind in appears_once:
                # seeing for the second time
                appears_once.remove(ind)
                appears_many.add(ind)
            elif ind not in appears_many:
                # first time we've seen this index
                appears_once.add(ind)
            # else seen twice or more - nothing to do
        uterms.append(uterm)

    # track post reduction indices
    post_reduce_appearances = {}

    # first pass to perform single term reductions
    for t, (term, uterm) in enumerate(zip(terms, uterms)):
        new = "".join(ix for ix in uterm if ix not in appears_once)
        if new != term:
            # do reduction
            f = functools.partial(
                get_func("einsum", backend), f"{term}->{new}"
            )
            lazy_temps[t] = Lazy(new, f, lazy_inputs[t])

        # collect duplicate terms
        post_reduce_appearances.setdefault(new, []).append(t)

    # second pass to perform multi term reductions
    for term, where in post_reduce_appearances.items():
        if len(term) == 0:
            # all the scalars -> multiply into smallest other term
            try:
                _, first, lz = min(
                    (len(lz.inds), t, lz)
                    for t, lz in enumerate(lazy_temps)
                    if (lz is not None) and (lz.inds != "")
                )
                args = (lz, *(lazy_temps[t] for t in where))
                rest = where
                inds = lz.inds
            except ValueError:
                # which may not exist... (all inputs are scalars)
                first, *rest = where
                args = tuple(lazy_inputs[t] for t in where)
                inds = ""

            lazy_temps[first] = Lazy(inds, multiply, *args)
            for t in rest:
                lazy_temps[t] = None

        elif len(where) > 1:
            args = [lazy_temps[t] for t in where]
            first, *rest = where
            lazy_temps[first] = Lazy(term, multiply, *args)
            for t in rest:
                lazy_temps[t] = None

    # finally filter out removed terms
    new_terms = []
    lazy_outputs = []
    for lz in lazy_temps:
        if lz is not None:
            new_terms.append(lz.inds)
            lazy_outputs.append(lz)

    def simplifier(arrays):
        for a, lz in zip(arrays, lazy_inputs):
            # inject arrays
            lz.args = (a,)
        # materialize the new terms
        return [lz.compute() for lz in lazy_outputs]

    # return new equation in same specifcation, and the simplifier
    if format == 0:
        new = terms, output
    elif format == 1:
        new = ",".join(new_terms), output
    else:  # format == 2,3:
        # we have to return output to guarantee einsum is still valid
        # e.g. 'ab,ab' after simplification requires 'ab->' not 'ab'
        new = ",".join(new_terms) + "->" + output

    return new, simplifier
