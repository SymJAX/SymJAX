import jax
import jax.numpy as np
import numpy as NP


def isdep(item):
    return isinstance(item, Variable) or isinstance(item, Placeholder)


class Op:
    """
    This class creates an Op object. An Op is a callable that encodes some
    internal computation in term of an evaluation function and its arguments.
    When an Op object is called with some arguments, it returns an actual node
    (Tensor) that represents the output value of the internal evaluation
    function computation given the arguments. Hence, its role is to provide a
    callable that will generate (at each call) a novel node (Tensor).

    Notes:
        This class is used to transform any function containing operations in
        jax.numpy into TheanoXLA nodes.

    Args:
        fn (callable): the function representing the computation to be
            performed on the given input arguments

    Attributes:
        fn (callable): the given internal evaluation function

    Examples:
        >>> def norm2py(x):
                return jax.numpy.sum(jax.numpy.square(x))
        >>> norm2 = Op(norm2py)
        >>> tensor = T.zeros((10, 10))
        >>> norm2(tensor)
        (Tensor, dtype=float32, shape=())
    """

    def __init__(self, fn, name='', docstring=None):
        """function that produces a new Op based on a given function"""
        self.fn = fn
        # set up the correct docstring
        if docstring is not None:
            self.__docstring__ = docstring
        else:
            if hasattr(fn, '__docstring__'):
                self.__docstring__ = fn.__docstring__

    def __call__(self, *args, _shape=None, _dtype=None, **kwargs):
        return Tensor(_eval=self.fn, args=args, kwargs=kwargs, shape=_shape,
                      dtype=_dtype)


class RandomOp(Op):
    """
    This class inherits form the :obj:`Op` object. Its role is to provide a
    callable that will generate (at each call) a novel node (RandomTensor)
    that can then be part of some computation graph.
    In particular this op is used instead of the :obj:`Op` when dealing with
    random functions taking :obj:`jax.random.PRNGKey` object as first argument.
    The callable will return a :obj:`RandomTensor` node.

    Notes:
        This class is used to transform the :mod:`jax.random` functions into
        TheanoXLA nodes.

    Args:
        msg (str): Human readable string describing the exception.
        code (:obj:`int`, optional): Error code.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    Examples:
        >>> normal = RandomOp(jax.random.normal)
        >>> normal((3, 3))
        (RandomTensor : dtype=float32, shape=(3, 3))
    """

    def __call__(self, *args, _shape=None, _dtype=None, seed=None, name='',
                 descr='', **kwargs):
        # the seed determines the value of the realisation, if not given, one
        # is assigned based on a cumulative seed tracking. That is, each new
        # random variable instanciated without a seed with get the current
        # seed number which is then incremented
        if seed is None:
            # if first time, the accumulator is created
            # else, we increment it and use it as current seed
            if 'seed_acc' not in globals():
                globals()['seed_acc'] = 0
            else:
                globals()['seed_acc'] += 1
            seed = globals()['seed_acc']
        tensor =  RandomTensor(_eval=self.fn, args=args, kwargs=kwargs,
                               shape=_shape, dtype=_dtype, seed=seed,
                               name=name, descr=descr)
        return tensor





class Tensor:

    def __init__(self, _eval, args=[], kwargs={}, shape=None,
                 dtype=None, name=''):

        self.args, self.kwargs = args, kwargs
        self.name = 'name=' + name + ', ' if name != '' else ''
        self._eval = _eval
        self.eval_value = None

        # set shape and dtype
        if shape is None or dtype is None:
            tree = jax.eval_shape(_eval, *self.args, **self.kwargs)
            self.shape, self.dtype = tree.shape, tree.dtype
        else:
            self.shape = shape
            self.dtype = dtype

        # set dependencies
        all_tensors = list(args) + list(kwargs.values())
        current_dependencies = [arg for arg in all_tensors if isdep(arg)]
        above_dependencies = sum([arg.all_dependencies for arg in all_tensors
                                if hasattr(arg,'all_dependencies')], [])
        self.all_dependencies = list(set(current_dependencies +
                                         above_dependencies))

    def __repr__(self):
        return '(Tensor' + self.name + ', dtype=' + str(self.dtype) + \
               ', shape='+str(self.shape) + ')'

    def __str__(self):
        return self.__repr__()

    def reset_value(self, propagate=False):
        self.eval_value = None
        if propagate:
            for item in self.args:
                if hasattr(item, 'reset_value'):
                    item.reset_value(True)
            for item in self.kwargs.values():
                if hasattr(item, 'reset_value'):
                    item.reset_value(True)

    def get(self, ins=dict(), force=False):
        # argument list
        if self.eval_value is not None and not force:
            return self.eval_value
        args = list()
        for arg in self.args:
            if hasattr(arg, 'get'):
                args.append(arg.get(ins, force))
            else:
                args.append(arg)
        # kwarg dictionnary
        kwargs = dict()
        for key, item in zip(self.kwargs.keys(), self.kwargs.values()):
            if hasattr(item, 'get'):
                kwargs.update({key: item.get(ins, force)})
            else:
                kwargs.update({key: item})
        self.eval_value = self._eval(*args, **kwargs)
        return self.eval_value



class RandomTensor(Tensor):
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
    def __init__(self, _eval, seed, shape=None, dtype=None, args=(), kwargs={},
                 name='', descr=''):
        self.seed = seed
        self.descr = descr
        key = jax.random.PRNGKey(seed)
        if type(args) == list:
            args = tuple(args)
        # the shape and dtype argument must be given as they are not traceable
        # we thus do a dummy evaluation a priori to get htem if not given
        if shape is None or dtype is None:
            dummy = _eval(*((key,)+args), **kwargs)
            shape, dtype = dummy.shape, dummy.dtype
        super().__init__(_eval=_eval, args=(key,)+args, kwargs=kwargs,
                         shape=shape, dtype=dtype, name=name)

    def __repr__(self):
        return '(RandomTensor ' + self.descr + ': ' + self.name + 'dtype=' + str(self.dtype) + \
               ', shape='+str(self.shape) + ')'

    def get(self, ins=dict(), force=False):
        # argument list
        if self.eval_value is not None and not force:
            return self.eval_value
        args = list()
        if 'rng' in ins:
            key = jax.random.PRNGKey(self.seed+ins['rng'])
        else:
            key = jax.random.PRNGKey(self.seed)
        args.append(key)
        for arg in self.args[1:]:
            if hasattr(arg, 'get'):
                args.append(arg.get(ins, force))
            else:
                args.append(arg)
        # kwarg dictionnary
        kwargs = dict()
        for key, item in zip(self.kwargs.keys(), self.kwargs.values()):
            if hasattr(item, 'get'):
                kwargs.update({key: item.get(ins, force)})
            else:
                kwargs.update({key: item})
        self.eval_value = self._eval(*args, **kwargs)
        return self.eval_value





class SubTuple(Tensor):

    def __init__(self, shape, dtype, index, parent):
        self.parent = parent
        self.index = index
        self.all_dependencies = parent.all_dependencies
        if not hasattr(self, 'name'):
            self.name = ''
        super().__init__(None, shape=shape, dtype=dtype)

    def reset_value(self, propagate=False):
        self.parent.reset_value(propagate)

    def get(self, ins=dict(), force=False):
        return self.parent.get(ins, force)[self.index]



class List(list):

    def __init__(self, _eval, shapes, dtypes, args=[], kwargs={}):

        self.args = args
        self.kwargs = kwargs
        all_tensors = list(args) + list(kwargs.values())
        above_dependencies = sum([arg.all_dependencies for arg in all_tensors
                                  if hasattr(arg,'all_dependencies')], [])
        current_dependencies = [arg for arg in all_tensors if isdep(arg)]
        self.all_dependencies = list(set(current_dependencies +
                                         above_dependencies))

        self._eval = _eval
        self.shapes = shapes
        self.dtypes = dtypes
        self.name = ''
        items = [SubTuple(shape, dtype, i, self)
                 for shape, dtype, i in zip(self.shapes, self.dtypes,
                                            range(len(shapes)))]
        self.eval_value = None
        super().__init__(items)

    def reset_value(self, propagate):
        self.eval_value = None
        if propagate:
            for item in self.args:
                if hasattr(item, 'reset_value'):
                    item.reset_value(True)
            for item in self.kwargs.values():
                if hasattr(item, 'reset_value'):
                    item.reset_value(True)


    def get(self, ins=dict(), force=False):
        if self.eval_value is not None and not force:
            return self.eval_value

        # arg list
        args = list()
        for arg in self.args:
            if hasattr(arg, 'get'):
                args.append(arg.get(ins))
            else:
                args.append(arg)

        # kwarg dictionnary
        kwargs = dict()
        for key, item in zip(self.kwargs.items()):
            if hasattr(item, 'get'):
                kwargs.update({key: item.get(ins)})
            else:
                kwargs.update({key: item})

        self.eval_value = self._eval(*args, **kwargs)
        return self.eval_value


class Variable(Tensor):

    def __init__(self, value, name='', trainable=True):
        self.trainable = trainable
        if not isinstance(value, jax.interpreters.xla.DeviceArray):
            self.value = np.array(value)
            if NP.isscalar(value):
                shape = ()
                dtype = type(value)
            else:
                shape = NP.shape(value)
                dtype = NP.dtype(value)
        else:
            self.value = value
            shape = value.shape
            dtype = value.dtype
        name = 'Variable: name='+name+', trainable='+str(trainable)
        super().__init__(None, shape=shape, dtype=dtype, name=name)

    def get(self, args, force=False):
        if self in args:
            return args[self]
        else:
            return self.value


class Placeholder(Tensor):

    def __init__(self, shape, dtype, name=''):
        name = 'Placeholder ' + name
        super().__init__(None, shape=shape, dtype=dtype, name=name)

    def get(self, args, force=False):
        assert self in args
        return args[self]
