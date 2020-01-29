import jax
import jax.numpy as jnp
from jax.random import _is_prng_key
import numpy
import inspect
import copy
from functools import wraps


def create_generic_class(func):
    name = func.split('.')[-1]
    exec('global {}\nclass {}(Op):\n\tpass\nadd_fn({})({})'.format(name, name,
                                                                   name, func))


def add_fn(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        setattr(cls, 'fn', staticmethod(wrapper))
        return func  # returning func means func can still be used normally
    return decorator


def add_method(cls):
    # important we keep the self inside the function call !
    def decorator(func, name=''):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        if name == '':
            setattr(cls, func.__name__, wrapper)
        else:
            setattr(cls, name, wrapper)
        return func  # returning func means func can still be used normally
    return decorator


def args_formatting(args, extra_args, indices):
    """ utility function to be used in the Tensor class to correctly join the
    args and extra_args based on the indices

    Parameters:
    -----------

    args: List

    extra_args: List

    indices: List of binary values
        the indices (one per element) to join args and extra_args in the correct
        order

    """
    output = ()
    arg_iterator = iter(args)
    extra_arg_iterator = iter(extra_args)
    for i in indices:
        #        print(i, args, extra_args)
        if i:
            output += (next(extra_arg_iterator),)
        else:
            output += (next(arg_iterator),)
    return output


def reset(item):
    if isinstance(item, list) or isinstance(item, tuple):
        [reset(i) for i in item]
    elif hasattr(item, 'eval_value'):
        item.eval_value = None
    if hasattr(item, 'kwargs'):
        for i in item.kwargs.values():
            reset(i)


def getroots(item, roots=[]):
    if isinstance(item, list) or isinstance(item, tuple):
        return roots + sum([getroots(i, roots) for i in item], [])
    elif hasattr(item, 'roots'):
        return roots + item.roots
    else:
        return []


def get(item, tracker):
    if isinstance(item, list) or isinstance(item, tuple):
        current = [get(i, tracker) for i in item]
        return current
    elif item in tracker:
        return tracker[item]
    elif hasattr(item, 'get'):
        item.get(tracker)
        return tracker[item]
    else:
        return item


def isvar(item):
    """ check whether an item (possibly a nested list etc) contains a variable
    (any subtype of Tensor) """
    # in case of nested lists/tuples, recursively call the function on it
    if isinstance(item, list) or isinstance(item, tuple):
        return numpy.sum([isvar(value) for value in item])
    # otherwise cheack that it is a subtype of Tensor or a Tracer and not
    # a callable
    else:
        cond1 = isinstance(item, Tensor)
        cond2 = isinstance(item, jax.interpreters.partial_eval.JaxprTracer)
        cond3 = not callable(item)
        return (cond1 or cond2) and cond3


class Tensor:

    def __init__(self, shape, dtype, roots=[], copyof=None):
        self.copyof = copyof
        self.roots = roots
        self._shape = tuple(shape)
        self._dtype = dtype

    def __repr__(self):
        return '(Tensor: shape={}, dtype={})'.format(self.shape, self.dtype)

    def __str__(self):
        return self.__repr__()

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def T(self):
        pass

    def get(self, tracker=None):
        """ this method implements only the case where the tensor is a copy
            of another one such as a variable etc. In this case we simply refer
            to the original get method. Otherwise, there should never be a call
            of get on a Tensor but always on an Op etc"""
        if self.copyof is not None:
            output = self.copyof.get(tracker)
            tracker[self] = output
            return output


class Op(Tensor):
    """an Op generates a Tensor object obtained from a function"""

    name = ''

    def __init__(self, *args, roots=[], **kwargs):
        self.kwargs = kwargs
        self.args = args
        self.print_name = 'name=' + self.name

        # set roots
        roots = getroots([i for i in kwargs.values()] + list(args)) + roots
        roots = list(set(roots))

        # now use the builin function to infer shape and dtype given a
        # lambda jax function, we need to remove the static arguments first
        self.extra_kwargs = {}
        for name, arg in list(self.kwargs.items()):
            if name == 'key' or not isvar(arg):
                self.extra_kwargs.update({name: arg})
                del self.kwargs[name]

        indices = list()
        for i, arg in enumerate(self.args):
            if not isvar(arg):
                indices.append(1)
            else:
                indices.append(0)

        self.extra_args = [arg for i,
                           arg in enumerate(self.args) if indices[i]]
        self.args = [arg for i, arg in enumerate(self.args) if indices[i] == 0]
        self.indices = indices

        tree = jax.eval_shape(lambda *args, **kwargs: self.fn(*args_formatting(args, self.extra_args, self.indices), **kwargs, **self.extra_kwargs), *self.args, **self.kwargs)
        shape, dtype = tree.shape, tree.dtype

        super().__init__(shape, dtype, roots)

    def __repr__(self):
        return '(Op: ' + self.fn.__name__ + ', shape=' + str(self.shape) +\
               ', dtype=' + str(self.dtype) + ')'

    def __str__(self):
        return self.__repr__()

    def get(self, tracker=None):
        if tracker is None:
            tracker = dict()
        if self in tracker:
            return tracker[self]

        # evaluate the function kwargs as explicit jax arrays
        kwargs = dict()
        for name, var in list(self.kwargs.items()):
            kwargs.update({name: get(var, tracker)})
        args = [get(var, tracker) for var in self.args]
        tracker[self] = self.fn(*args_formatting(args, self.extra_args, self.indices), **kwargs, **self.extra_kwargs)
        return tracker[self]


class RandomOp(Op):
    """
    This class creates a :obj:`Tensor` object that given a function (see below)
    and its inputs can be used as a Node in the graph construction. This class
    is specialized to deal with random functions, if the function does not
    take a jax.PRNGKey as first argument, then it should not be used.

    Notes:
        This class is not meant to be used by the user. To create your own
        callable node, see :class:`RandomOp`.

    Args:
        msg (str): Human readable string describing the exception.
        code (:obj:`int`, optional): Error code.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    Examples:
        >>> node = RandomTensor(jax.random.bernoulli, 0, args=(0.5, (3, 3)))
        >>> print(node)
        (RandomTensor : name=custom, dtype=bool, shape=(3, 3))
        >>> node + 2
        (Tensor, dtype=int32, shape=(3, 3))
    """

    def __init__(self, *args, seed=None, **kwargs):
        if seed is None:
            seed = numpy.random.randint(0, 2147483647)
        self.seed = seed
        key = jax.random.PRNGKey(seed)
        super().__init__(key, *args, **kwargs, roots=[self])

    def __repr__(self):
        return '(RandomTensor: ' + self.print_name + ', dtype='\
            + str(self.dtype) + ', shape=' + str(self.shape) + ')'

    def get(self, tracker=None):
        if tracker is None:
            tracker = dict()
        elif self in tracker:
            return tracker[self]
        # argument list
        if 'rng' in tracker:
            key = jax.random.PRNGKey(self.seed + tracker['rng'])
            self.extra_args[0] = key
        return super().get(tracker)


class SubTensor(Tensor):

    def __init__(self, shape, dtype, index, parent, roots, name=''):
        self.parent = parent
        self.index = index
        super().__init__(shape, dtype, roots=roots)

    def get(self, tracker=None):
        return self.parent.get(tracker)[self.index]


class Tuple(tuple):

    def __new__(cls, fn, args=[], kwargs={}):
        trees = jax.eval_shape(lambda *a, **b: fn(*a, **b), *args,
                               **kwargs)
        items = [SubTensor(t.shape, t.dtype, i, None, roots=[])
                 for i, t in enumerate(trees)]

        return super(Tuple, cls).__new__(cls, tuple(items))

    def __init__(self, fn_or_list, args=[], kwargs={}):

        self.args = args
        self.kwargs = kwargs
        self.fn = fn_or_list
        self.name = ''
        # set roots
        roots = list()
        for value in kwargs.values():
            if hasattr(value, 'roots'):
                roots += value.roots
        for value in args:
            if hasattr(value, 'roots'):
                roots += value.roots

        roots = list(set(roots))
        for item in self:
            item.parent = self
            item.roots = roots

        self.args, self.kwargs = args, kwargs

    def get(self, tracker=None):
        if tracker is None:
            tracker = dict()
        if self in tracker:
            return tracker[self]

        # kwarg dictionnary
        args = list()
        for var in self.args:
            args.append(get(var, tracker))

        kwargs = dict()
        for name, var in self.kwargs.items():
            kwargs.update({name: get(var, tracker)})

        # we add the list object itself into the dictionnary first
        tracker[self] = tuple(self.fn(*args, **kwargs))

        # then we add each element of the list into the dictionnary
        for i in range(len(self)):
            tracker[self[i]] = tracker[self][i]

        return tracker[self]


class Variable(Tensor):
    """variable that is a standalone persistent tensor. this tensor
    can be updated and differentiated.

    Parameters:
    -----------

        value_or_fn: array or initializer
            the value given as a numpy array or an initializer which
            takes as input the shape and can be type casted afterward
            via numpy.cast

        shape: tuple (optional)
            the shape of the variable, used only if the value_or_fn is an
            initializer

        dtype: dtype (optional)
            the dtype of the variable, used only if the value_or_fn is an
            initializer

        name: str (optional)
            the name of the variable, there is no test of name duplication

        trainable: bool
            whether the variable is trainable or not. It is set as an
            attribute and can be accessed.
    """

    def __init__(self, value_or_fn, shape=None, dtype=None,
                 name='', trainable=True):
        self.trainable = trainable
        self.name = name
        if callable(value_or_fn):
            if dtype is not None:
                value = value_or_fn(shape).astype(dtype)
            else:
                value = value_or_fn(shape).astype(dtype)
            self.value = jnp.asarray(value)
            self.init_value = (value_or_fn, shape)
        else:
            self.value = jnp.asarray(value_or_fn)
            self.init_value = copy.deepcopy(self.value)
        if hasattr(self.value, 'shape'):
            shape = self.value.shape
            dtype = self.value.dtype
        else:
            shape = ()
            dtype = type(value)

        super().__init__(shape, dtype, roots=[self])

    def reset(self):
        """reset the value of the variable based on the initial one, whether
        it was an array or initializer. If it was a random initializer,
        nothing guarantees that the reset will give back the original value
        as opposed to the array case
        """

        if isinstance(self.value, tuple):
            value = self.init_value[0](self.init_value[1])
        else:
            value = self.init_value
        self.value = jnp.asarray(value).astype(self.dtype)

    def __repr__(self):
        return '(Variable: ' + self.name + 'dtype=' + str(self.dtype) + \
               ', shape=' + str(self.shape) + ', trainable=' + \
            str(self.trainable) + ')'

    def get(self, tracker):
        if self not in tracker:
            tracker[self] = self.value
        return tracker[self]


class Placeholder(Tensor):
    """placeholder is an input to the computational graph that takes outside
    values. That is, it is an input gate to feed data into a computational
    graph as opposed to for example variables which are living in memory and
    are not fed externally.

    Parameters:
    -----------

        shape: tuple
            the shape of the placeholder

        dtype: dtype
            the dtype of the placeholder

        name: str (optional)
            the name of the variable, there is no test of name duplication
    """

    def __init__(self, shape, dtype, name=''):
        self.name = name
        super().__init__(shape, dtype, roots=[self])

    def __repr__(self):
        return '(Placeholder: ' + self.name + 'dtype=' + str(self.dtype) + \
               ', shape=' + str(self.shape) + ')'

    def get(self, tracker):
        if self not in tracker:
            raise ValueError(' no value given for placeholder {}'.format(self))
        return tracker[self]


def placeholder_like(item, name=''):
    return Placeholder(item.shape, item.dtype, name=name)


def theanofn_to_jaxfn(*args, _fn, **kwargs):

    # treat the args
    pargs = list()
    for arg in args:
        pargs.append(placeholder_like(arg))

    # treat the kwargs
    pkwargs = dict()
    for name, var in kwargs.items():
        pkwargs[name] = placeholder_like(var)
    output = _fn(*pargs, **pkwargs)
    feed_dict = list(zip(pargs, args)) + list(zip(pkwargs.values(),
                                                  kwargs.values()))
    return output.get(dict(feed_dict))
