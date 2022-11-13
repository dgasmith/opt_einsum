import operator
import functools
import collections

from .backends import get_func
from .parser import find_output_str


def unique(it):
    return dict.fromkeys(it).keys()


def multiply(*xs):
    """Functional version of ``x[0] * x[1] * ...``."""
    return functools.reduce(operator.mul, xs)


class Lazy:
    """A tiny lazy computation class."""

    __slots__ = ("inds", "f", "args")

    def __init__(self, inds, f=None, *args):
        self.inds = inds
        self.f = f
        self.args = args

    def compute(self):
        """Compute the result of this lazy node, first computing all lazy
        children.
        """
        return self.f(
            *(
                arg.compute() if isinstance(arg, Lazy) else arg
                for arg in self.args
            )
        )

    def __repr__(self):
        return f"Lazy('{self.inds}', {self.f}, {self.args})"


class SimplifyExpression:

    def __init__(self, lazy_inputs, lazy_outputs):
        self.lazy_inputs = lazy_inputs
        self.lazy_outputs = lazy_outputs

    def __call__(self, arrays):
        for a, lz in zip(arrays, self.lazy_inputs):
            # inject arrays
            lz.args = (a,)
        # materialize the new terms
        return tuple(lz.compute() for lz in self.lazy_outputs)

    def __repr__(self):
        return (
            f"SimplifyExpression('"
            f"{','.join(lz.inds for lz in self.lazy_inputs)}->"
            f"{','.join(lz.inds for lz in self.lazy_outputs)}')"
        )


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

    # initial maps of where indices and whole terms appear
    ind_appearances = collections.defaultdict(set)
    term_appearances = collections.defaultdict(set)
    queue = []
    for t, term in enumerate(terms):
        queue.append(t)
        term_appearances[term].add(t)
        for ix in term:
            ind_appearances[ix].add(t)
    for ix in output:
        ind_appearances[ix].add(-1)

    # want to make everything lazy
    lazy_inputs = [Lazy(term, lambda x: x) for term in terms]
    lazy_temps = dict(enumerate(lazy_inputs))

    # flag to check if we should keep looping
    should_run = True
    while should_run:
        should_run = False

        # XXX: only iterate over t we need to check, not everything?
        for t, lz in lazy_temps.items():
            term = lz.inds
            reduced = ""
            # unique call takes care of tracing
            for ix in unique(term):
                if len(ind_appearances[ix]) == 1:
                    # ind guaranteed only appears here - remove
                    del ind_appearances[ix]
                else:
                    # keep after reduction
                    reduced += ix

            if reduced != term:
                # perform reduction (summing and/or tracing)
                f = functools.partial(
                    get_func("einsum", backend), f"{term}->{reduced}"
                )
                # replace with new reduce lazy node
                lazy_temps[t] = Lazy(reduced, f, lz)

                # update maps
                t_aps = term_appearances[term]
                if len(t_aps) == 1:
                    # entry would be empty set
                    del term_appearances[term]
                else:
                    t_aps.remove(t)
                term_appearances[reduced].add(t)

        # check multi term reductions
        for term, where in tuple(term_appearances.items()):
            if term == '':
                # all the scalars
                try:
                    # try to multiply into smallest term
                    _, first, lz = min(
                        (len(lz.inds), t, lz)
                        for t, lz in lazy_temps.items()
                        if (lz is not None) and (lz.inds != "")
                    )
                    inds = lz.inds
                    args = (lz, *map(lazy_temps.pop, where))
                except ValueError:
                    # which may not exist... (if all inputs are scalars)
                    # multiply into a single scalar instead
                    first = min(where)
                    inds = ""
                    args = map(lazy_temps.pop, where)
                lazy_temps[first] = Lazy(inds, multiply, *args)
                term_appearances.pop(term)

            elif len(where) > 1:
                # hadamard deduplication
                first = min(where)
                args = []
                rest = set()
                for t in where:
                    if t == first:
                        # don't pop the first to maintain order
                        args.append(lazy_temps[t])
                    else:
                        args.append(lazy_temps.pop(t))
                        rest.add(t)
                lazy_temps[first] = Lazy(term, multiply, *args)
                term_appearances[term] = {first}
                for ix in term:
                    ind_appearances[ix] -= rest

                # we only need to run again if we did hadamard deduplication
                should_run = True

    # get the final terms and lazy nodes
    new_terms = []
    lazy_outputs = []
    for lz in lazy_temps.values():
        new_terms.append(lz.inds)
        lazy_outputs.append(lz)

    # return new equation in same specifcation, and the simplifier
    if format == 0:
        new = new_terms, output
    elif format == 1:
        new = ",".join(new_terms), output
    else:  # format == 2,3:
        # we have to return output to guarantee einsum is still valid
        # e.g. 'ab,ab' after simplification requires 'ab->' not 'ab'
        new = ",".join(new_terms) + "->" + output

    return new, SimplifyExpression(lazy_inputs, lazy_outputs)
