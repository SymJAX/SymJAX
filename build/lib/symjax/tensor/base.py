import re
from functools import wraps

import jax
import jax.numpy as jnp
import numpy

import symjax


def _add_method(cls):
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


def _args_formatting(args, extra_args, indices):
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


def get(item, tracker=None, givens=None, branches=None):
    if tracker is None:
        tracker = {}
    if givens is None:
        givens = {}
    if branches is None:
        branches = {}

    if isinstance(item, list):
        return [get(i, tracker, givens, branches) for i in item]
    elif isinstance(item, tuple):
        return tuple([get(i, tracker, givens, branches) for i in item])

    if isinstance(item, Tensor):
        # if the item is already in tracker, we might just return the already
        # computed one unless it has to be altered by clone, in that case
        # the actual output value will not be the one from the tracker and 
        # we thus have to specialize the tracker for each ''branch''
        current_givens = {**givens, **item._givens}
        minimal = get_connected(item, current_givens.keys())
        minimal_givens = dict([m for m in current_givens.items() if m[0] in minimal])
        if item in minimal_givens:
            new_givens = dict([m for m in minimal_givens.items() if m[0] != item])
            return get(minimal_givens[item], tracker, new_givens, branches)
        if len(minimal_givens) == 0:
            # if this branch is unchanged, return directly the already
            # computed one
            if item in tracker:
                return tracker[item]
            tracker[item] = item._get(tracker, {}, {})
            return tracker[item]

        # otherwise we specialize
        names = ['{}{}->{}{}'.format(m.scope, m.name, current_givens[m].scope,
                                     current_givens[m].name) for m in minimal]
        names.sort()
        name = '_'.join(names)
        if item in branches:
            if name in branches[item]:
                return branches[item][name]
            branches[item][name] = item._get(tracker, minimal_givens, branches)
        else:
            branches[item] = {name: item._get(tracker, minimal_givens, branches)}
        return branches[item][name]
    else:
        return item


def get_connected(item, parents, _minimal=None):
    """
    Utility function returning the list of connected guys from the provided one

    Parameters
    ----------

    item: Tensor or any value

    parents: list of things

    Returns

    connected: list
    :param _minimal:

    """
    if len(parents) == 0:
        return []
    if _minimal is None:
        _minimal = []
    if hasattr(item, 'args'):
        for arg in item.args:
            if arg in parents and arg not in _minimal:
                _minimal.append(arg)
            _minimal = get_connected(arg, parents, _minimal)
    if hasattr(item, 'kwargs'):
        for arg in item.kwargs.values():
            if arg in parents and arg not in _minimal:
                _minimal.append(arg)
            _minimal = get_connected(arg, parents, _minimal)
    if item in parents and item not in _minimal:
        _minimal.append(item)
    return _minimal


def isvar(item):
    """ check whether an item (possibly a nested list etc) contains a variable
    (any subtype of Tensor) """
    # in case of nested lists/tuples, recursively call the function on it
    if isinstance(item, slice):
        return False
    elif isinstance(item, list) or isinstance(item, tuple):
        return numpy.sum([isvar(value) for value in item])
    # otherwise cheack that it is a subtype of Tensor or a Tracer and not
    # a callable
    else:
        cond1 = isinstance(item, Tensor)
        #        cond2 = isinstance(item, jax.interpreters.partial_eval.JaxprTracer)
        cond3 = callable(item)
        return cond1 and not cond3  # (cond1 or cond2) and cond3


class Tensor:
    __array_priority__ = 1000

    def __init__(self, shape, dtype, roots=[], copyof=None, name=None):
        self.copyof = copyof
        self._roots = roots
        self._shape = tuple(shape)
        self._dtype = dtype
        self._givens = {}
        if name is not None:
            assert '/' not in name
            self._name = name
        else:
            self._name = 'unnamed'
        symjax.current_graph().add(self)

    def __repr__(self):
        return '(Tensor: name={}, shape={}, dtype={})'.format(self.name,
                                                              self.shape, self.dtype)

    def __str__(self):
        return self.__repr__()

    @property
    def name(self):
        return self._name

    def _set_name(self, new_name):
        self._name = new_name

    @property
    def args(self):
        if self.copyof is not None:
            return self.copyof.args
        elif hasattr(self, '_args'):
            return self._args
        else:
            return []

    @property
    def roots(self):
        if self.copyof is not None:
            return self.copyof.roots
        elif hasattr(self, '_roots'):
            return self._roots
        else:
            return []

    @property
    def kwargs(self):
        if self.copyof is not None:
            return self.copyof.kwargs
        elif hasattr(self, '_kwargs'):
            return self._kwargs
        else:
            return {}

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self.shape)

    def _get(self, tracker, givens, branches):
        """ this method implements only the case where the tensor is a copy
            of another one such as a variable etc. In this case we simply refer
            to the original get method. Otherwise, there should never be a call
            of get on a Tensor but always on an Op etc"""
        if self.copyof is not None:
            return self.copyof._get(tracker, givens, branches)

    def clone(self, givens):
        for g in givens:
            assert isinstance(givens[g], Tensor)
        new_object = Tensor(self.shape, self.dtype, self.roots, copyof=self,
                            name=self.name + '_clone')
        new_object._givens = givens
        return new_object

    def _check_tracker(self, tracker):
        if tracker is None:
            return
        for i in tracker:
            if isinstance(tracker[i], Tensor):
                RuntimeError("incorrect tracker value for {}".format(tracker[i]))


_numpy_signature_re = re.compile(r'^([\w., ]+=)?\s*[\w\.]+\(.*\)$')


def update_numpydoc(docstr, fun, op):
    '''Transforms the numpy docstring to remove references of
       parameters that are supported by the numpy version but not the JAX version'''

    # Some numpy functions have an extra tab at the beginning of each line,
    # If this function is one of those we remove this extra tab from all the lines
    if not hasattr(op, '__code__'):
        return docstr
    if docstr[:4] == '    ':
        lines = docstr.split('\n')
        for idx, line in enumerate(lines):
            lines[idx] = line.replace('    ', '', 1)
        docstr = '\n'.join(lines)

    begin_idx = docstr.find("Parameters")
    begin_idx = docstr.find("--\n", begin_idx) + 2
    end_idx = docstr.find("Returns", begin_idx)

    parameters = docstr[begin_idx:end_idx]
    param_list = parameters.replace('\n    ', '@@').split('\n')
    for idx, p in enumerate(param_list):
        param = p[:p.find(' : ')].split(", ")[0]
        if param not in op.__code__.co_varnames:
            param_list[idx] = ''
    param_list = [param for param in param_list if param != '']
    parameters = '\n'.join(param_list).replace('@@', '\n    ')
    return docstr[:begin_idx + 1] + parameters + docstr[end_idx - 2:]


def jax_wrap(func, insert_default_kwargs=True, doc_func=None, is_method=False):
    if doc_func is None:
        doc_func = func

    @wraps(doc_func)
    def op(*args, seed=None, **kwargs):
        # now we evaluate the shape from the jax built-in function

        # first we check if we are in a random function to be careful
        # with the key
        from . import random
        random_func = func in random._RANDOM_FUNCTIONS

        # if there is a name we remove it for now to use the jax tracer
        if 'name' in kwargs:
            op_name = kwargs['name']
            del kwargs['name']
        else:
            op_name = None

        # we need to remove the static arguments first
        # we first do it for the kwars
        static_kwargs = {}
        var_kwargs = {}
        for name, arg in list(kwargs.items()):
            if not isvar(arg):
                static_kwargs.update({name: arg})
            else:
                var_kwargs.update({name: arg})

        # we need to do the same for the args
        indices = list()
        for i, arg in enumerate(args):
            if not isvar(arg):
                indices.append(1)
            else:
                indices.append(0)
        static_args = [arg for i, arg in zip(indices, args) if i]
        var_args = [arg for i, arg in zip(indices, args) if not i]

        # this is just to get shape and dtype so we do not bother
        # to use the correct seed yet
        if random_func:
            key = jax.random.PRNGKey(0)
            static_args = [key] + static_args
            indices.insert(0, 1)

        # we need to define an abstract function that only takes as input the
        # non-static arguments, internally join them with the static ones
        # and return the output. This is because the jax shape inference
        # functions does not work with static arguments (such as the dimensions
        # of the transpose function)
        def abstract_func(*args, **kwargs):
            all_args = _args_formatting(args, static_args, indices)
            return func(*all_args, **kwargs, **static_kwargs)

        # now we evaluate the shape from the jax built-in function
        tree = jax.eval_shape(abstract_func, *var_args, **var_kwargs)

        # now we determine if it is an Op or a Tuple object based on the
        # infered shape
        if type(tree) == list or type(tree) == tuple:
            shapes = [t.shape for t in tree]
            dtypes = [t.dtype for t in tree]
            return Tuple(
                *args,
                _jax_function=func,
                _shapes=shapes,
                _dtypes=dtypes,
                name=op_name,
                **kwargs)
        elif random_func:
            shape, dtype = tree.shape, tree.dtype
            return RandomOp(
                *args,
                _jax_function=func,
                _shape=shape,
                _dtype=dtype,
                _seed=seed,
                name=op_name,
                **kwargs)
        else:
            shape, dtype = tree.shape, tree.dtype
            return Op(*args, _jax_function=func, _shape=shape, _dtype=dtype,
                      name=op_name, **kwargs)

    if not hasattr(func, '__doc__') or func.__doc__ is None:
        return op

    if doc_func is not None:
        sections = func.__doc__.split("\n\n")

        signatures = []
        summary = None
        for i in range(len(sections)):
            if _numpy_signature_re.match(sections[i]):
                signatures.append(sections[i])
            else:
                summary = sections[i].strip()
                break
        body = "\n\n".join(signatures + sections[i + 1:])
        body = update_numpydoc(body, func, op)
        desc = "ADDITION"
        docstr = (
            "{summary}\n\nLAX-backend implementation of :func:`{fun}`.\n"
            "{lax_description}Original docstring below.\n\n{body}"
                .format(summary=summary, lax_description=desc,
                        fun=func.__name__, body=body))

        op.__name__ = func.__name__
        op.__doc__ = docstr

    return op


def wrap_class(c, method_exceptions=None):
    class meta:
        def __new__(cls, *args, **kwargs):

            # the first part consists into reexpressing any possible symjax
            # input into a jax one to first evaluate the class creator and
            # derive from its the equivalent symjax computational graph that
            # would produce the same class attributes
            new_args = []
            new_kwargs = {}
            for i in range(len(args)):
                if isinstance(args[i], Tensor):
                    new_args.append(jnp.zeros(args[i].shape, dtype=args[i].dtype))
                else:
                    new_args.append(args[i])
            for i in kwargs:
                if isinstance(kwargs[i], Tensor):
                    new_kwargs[i] = jnp.zeros(kwargs[i].shape, dtype=kwargs[i].dtype)
                else:
                    new_kwargs[i] = kwargs[i]

            # now we check which attributes were added during the class
            # creation, those are the ones that will have to be obtained from
            # a symjax computational graph based on the class inputs
            attr_before = c.__dict__.keys()
            instance = c(*new_args, **new_kwargs)
            attr_after = instance.__dict__.keys()

            news = [i for i in attr_after if i not in attr_before]
            news = [n for n in news if isinstance(instance.__dict__[n], jax.interpreters.xla.DeviceArray)]

            # this function maps the class inputs to the creator generated
            # class attributes
            def function(*args, **kwargs):
                return [instance.__dict__[n] for n in news]

            init_op = jax_wrap(function)

            # we now allow our normal class creation to proceed
            obj = super().__new__(cls)
            obj._init_op = init_op
            obj._news = news

            # we also have to wrap all the methods
            method_exceptions = cls._method_exceptions or []
            for att in dir(instance):
                if att[:2] == '__' or att in method_exceptions:
                    continue
                if callable(getattr(instance, att)):
                    setattr(obj, att, jax_wrap(getattr(instance, att)))
            return obj

        def __init__(self, *args, **kwargs):
            attrs = self._init_op(*args, **kwargs)
            for n, a in zip(self._news, attrs):
                self.__dict__[n] = a

    meta._method_exceptions = method_exceptions

    return meta


class Op(Tensor):
    """an Op generates a Tensor object obtained from a function"""

    def __init__(self, *args, _jax_function, _shape, _dtype, _roots=[], name=None,
                 **kwargs):

        # save args and kwargs
        self._kwargs = kwargs
        self._args = args
        self.jax_function = _jax_function
        # set roots
        roots = list(set(getroots(list(kwargs.values()) + list(args)) + _roots))

        super().__init__(_shape, _dtype, roots, name=name)

    def __repr__(self):
        name = 'Op(name={}, shape={}, dtype={}, scope={})'
        return name.format(self.name, self.shape, self.dtype,
                           self.scope)

    def __str__(self):
        return self.__repr__()

    def _get(self, tracker, givens, branches):

        self._check_tracker(tracker)

        # evaluate the function kwargs as explicit jax arrays
        kwargs = dict([(name, get(var, tracker, givens, branches))
                       for name, var in self.kwargs.items()])
        args = [get(var, tracker, givens) for var in self.args]

        if isinstance(self, RandomOp):
            seed = self._seed or numpy.random.randint(0, 1000000)
            if 'rng' in tracker:
                key = jax.random.PRNGKey(seed + tracker['rng'])
            else:
                key = jax.random.PRNGKey(seed)
            return self.jax_function(key, *args, **kwargs)
        else:
            return self.jax_function(*args, **kwargs)


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

    def __init__(self, *args, _jax_function, _shape, _dtype, _seed, name, **kwargs):
        self._seed = _seed
        super().__init__(*args, _jax_function=_jax_function, _shape=_shape,
                         _dtype=_dtype, name=name, **kwargs)

    def __repr__(self):
        name = 'RandomTensor(Op={}, shape={}, dtype={})'
        return name.format(self.name, self.shape, self.dtype)


class TupleItem(Tensor):

    def __init__(self, shape, dtype, index, roots, name=''):
        self._parent = None
        self._index = index
        super().__init__(shape, dtype, roots=roots, name=name)

    def _get(self, tracker, givens, branches):
        return self._parent._get(tracker, givens, branches)[self._index]


class Tuple(tuple):

    def __new__(cls, *args, _jax_function, _shapes, _dtypes, name, **kwargs):
        roots = list(set(getroots(list(kwargs.values()) + list(args))))
        name = name or _jax_function.__name__
        items = [TupleItem(shape, dtype, i, roots=roots, name=name + '[{}]'.format(i))
                 for i, (shape, dtype) in enumerate(zip(_shapes, _dtypes))]
        return super(Tuple, cls).__new__(cls, tuple(items))

    def __init__(self, *args, _jax_function, _shapes, _dtypes, name, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.jax_function = _jax_function
        self.name = name

        # set the parent link with the inside items and set the roots too
        for item in self:
            item._parent = self

    def _get(self, tracker, givens, branches):
        # kwarg dictionnary
        args = [get(var, tracker, givens, branches) for var in self.args]
        kwargs = dict([(name, get(var, tracker, givens, branches))
                       for name, var in self.kwargs.items()])
        return tuple(self.jax_function(*args, **kwargs))


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

    def __init__(self, initializer, name='unnamed_variable', trainable=True, dtype=None):

        self.trainable = trainable
        self.initializer = initializer
        self._dtype = dtype

        super().__init__(self.value.shape, str(self.value.dtype), roots=[self], name=name)

    def reset(self):

        """reset the value of the variable based on the initial one, whether
        it was an array or initializer. If it was a random initializer,
        nothing guarantees that the reset will give back the original value
        as opposed to the array case
        """
        if isinstance(self.initializer, Tensor):
            self._value = get(self.initializer)
        else:
            self._value = numpy.array(self.initializer)

        if self._dtype is not None:
            self._value = self._value.astype(self._dtype)

    @property
    def value(self):

        """ utility function that takes the input and return
            the actual value. It deals with cases where the input
            was a function or not etc
        """
        if not hasattr(self, '_value'):
            self.reset()
        return self._value

    def update(self, update_value):

        """assign a new value for the variable"""

        self._value = get(update_value)

    def __repr__(self):
        name = 'Variable(name={}, shape={}, dtype={}, trainable={}, scope={})'
        return name.format(self.name, self.shape, self.dtype, self.trainable, self.scope)

    def _get(self, tracker, givens, branches):
        return self.value


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
        super().__init__(shape, dtype, roots=[self], name=name)

    def __repr__(self):
        return '(Placeholder: ' + self.name + ', dtype=' + str(self.dtype) + \
               ', shape=' + str(self.shape) + ')'

    def _get(self, tracker, givens, branches):
        raise ValueError(' no value given for placeholder {}'.format(self))


def placeholder_like(item, name=''):
    if item is None:
        return None
    elif type(item) == list or type(item) == tuple:
        return type(item)([placeholder_like(i) for i in item])
    else:
        return Placeholder(item.shape, item.dtype, name=name)


def match(l1, l2, output):
    if output is None:
        output = dict()
    if type(l1) == list or type(l1) == tuple:
        for a, b in zip(l1, l2):
            match(a, b, output)
    else:
        output.update({l1: l2})


def symjax_to_jax_fn(func):

    def newfn(*args, fn=func):
        pholders = placeholder_like(args)
        symjax_outputs = fn(*pholders)
        feed_dict = {}
        match(pholders, args, feed_dict)
        if None in feed_dict:
            del feed_dict[None]
        outputs = get(symjax_outputs, feed_dict)
        return outputs

    return newfn


def clone(tensor, givens):
    return tensor.clone(givens)
