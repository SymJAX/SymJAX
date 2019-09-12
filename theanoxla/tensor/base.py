import jax
import jax.numpy as np
import numpy
import inspect




def get_output_for(values, feed_dict, reset=True):
    print('in getoutputfor with', values, feed_dict)

    if type(values) != list and type(values) != tuple:
        no_list = True
        values = [values]
    else:
        no_list = False
    # reset the values. This is needed as we do not force the value
    # computation below, and values might already have been computed with
    # some tracer to evaluate shape etc ... hence we force a full
    # graph evaluation again
    if reset:
        print('resetting')
        for value in values:
            print(value.eval_value)
            if hasattr(value, 'reset_value'):
                value.reset_value(True)
        for value in values:
            print(value.eval_value)

    if 'rng' not in feed_dict:
        if '_rng' not in globals():
            globals()['_rng'] = 0
        if reset:
            globals()['_rng'] += 1
        feed_dict['rng'] = globals()['_rng']

    outputs = list()
    for value in values:
        outputs.append(get(value, feed_dict))
    return outputs if not no_list else outputs[0]

def reset(item):
    if type(item) == list or type(item) == tuple:
        [reset(i) for i in item]
    elif hasattr(item, 'eval_value'):
        item.eval_value = None
    if hasattr(item, 'kwargs'):
        for i in item.kwargs.values():
            reset(i)


def get(item, tracker):
    if item in tracker:
        return tracker[item]
    elif type(item) == list or type(item) == tuple:
        current = [get(i, tracker) for i in item]
        return current
    elif hasattr(item, 'get'):
        item.get(tracker)
        return tracker[item]
    else:
        return item


def isdep(item):
    v = isinstance(item, Variable)
    p = isinstance(item, Placeholder)
    return p or v


def isvar(item):
    if type(item) == list or type(item) == tuple:
        return numpy.sum([isvar(value) for value in item])
    else:
        cond1 = isinstance(item, Tensor)
        cond2 = type(item) == jax.interpreters.partial_eval.JaxprTracer
        cond3 = not callable(item)
        return cond1 or cond2 or cond3


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
        self.name = name
#        # set up the correct docstring
#        if docstring is not None:
#            self.__docstring__ = docstring
#        else:
#            if hasattr(fn, '__docstring__'):
#                self.__docstring__ = fn.__docstring__

    def __call__(self, *args, _shape=None, _dtype=None, **kwargs):
        return Tensor(self.fn, args=args, kwargs=kwargs,
                      name='Op(' + self.name + ')', shape=_shape, dtype=_dtype)


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

    def __init__(self, fn_or_tensor, args=[], kwargs={}, shape=None,
                 dtype=None, name='', all_dependencies=[]):

        self.kwargs = kwargs
        self.is_fn = callable(fn_or_tensor)
        self.name = name
        self.print_name = 'name=' + name + ', ' if name != '' else ''

        if self.is_fn:
            # for convenience we only deal with kwargs, and thus transform
            # any given are into a kwarg based on the function signature
            signature = list(inspect.signature(fn_or_tensor).parameters.keys())
            for arg, name in zip(args, signature[:len(args)]):
                self.kwargs.update({name: arg})
            kwargs = self.kwargs.copy()
        self.fn = fn_or_tensor

        # set shape and dtype
        if (shape is None or dtype is None) and not isinstance(fn_or_tensor, Tensor):
            # if we have to use the automatic shape evaluation then we have to
            # take care of some problematic cases happening when a parameter
            # such as the shape in the reshape function is given. In fact, when
            # trying to infer the shape, each input ( arg or kwarg) is given as
            # a tracer to monitor internal computations. But those parameters
            # are not the ones that support such assignments and we thus have
            # to infer the shape using a slightly tweaked function

            # get the function signature (inputs and keywords)
            signature = list(inspect.signature(self.fn).parameters.keys())

            # to keep track of the removed arguments that won't be part of the
            # inputs of the tweaked function
            # then take care of the kwargs
            extra_kwargs = {}
            for name, arg in list(kwargs.items()):
                if not isvar(arg):
                    extra_kwargs.update({name: arg})
                    del kwargs[name]
            tree = jax.eval_shape(lambda **b: self.fn(**b, **extra_kwargs),
                                  **kwargs)
            self.shape, self.dtype = tree.shape, tree.dtype

        elif shape is not None and dtype is not None:
            self.shape = shape
            self.dtype = dtype

        else:
            assert not self.is_fn
            self.shape = fn_or_tensor.shape
            self.dtype = fn_or_tensor.dtype


    def __repr__(self):
        return '(Tensor: ' + self.print_name + 'dtype=' + str(self.dtype) + \
               ', shape='+str(self.shape) + ')'

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.shape[0] if self.ndim > 0 else 0

    @property
    def ndim(self):
        return len(self.shape)


    def get(self, tracker=dict()):
        if self in tracker:
            return tracker[self]

        # if the value given is already a tensor, delegates the
        # computation to its own method
        if not self.is_fn:
            return self.fn.get(tracker)

        # evaluate the function kwargs as explicit jax arrays
        kwargs = dict()
        for name, var in self.kwargs.items():
#            if name == 'true_fun' or name == 'false_fun':
#                arg = self.kwargs[name[:-3]+'operand']
#                argv = get(arg, ins)
#                feed_dict = dict(zip(arg, argv))
#                if hasattr(var, 'get'):
#                    kwargs.update({name: lambda args: get_output_for(var, dict(zip(arg, args)))})
#                else:
#                    kwargs.update({name: lambda args: var})
#                print(name, 'output',var, argv, kwargs[name](argv))
#                arg = self.kwargs['true_operand']
#                argv = get(arg, ins)
#                print('TRUE output', argv, kwargs['true_fun'](argv))
#            else:
            print(name, var)
            kwargs.update({name: get(var, tracker)})
#        print('function and kwargs', self.fn, kwargs)
        tracker[self] = self.fn(**kwargs)
        return tracker[self]






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
        super().__init__(_eval, args=(key,)+args, kwargs=kwargs,
                         shape=shape, dtype=dtype, name=name)

    def __repr__(self):
        return '(RandomTensor: ' + self.descr + self.print_name + 'dtype=' + str(self.dtype) + \
               ', shape='+str(self.shape) + ')'

    def get(self, tracker=dict()):
        if self in tracker:
            return tracker[self]
        # argument list
        if 'rng' in tracker:
            key = jax.random.PRNGKey(self.seed+tracker['rng'])
        else:
            key = jax.random.PRNGKey(self.seed)

        # kwarg dictionnary
        kwargs = dict()
        print(self.kwargs)
        for name, var in self.kwargs.items():
            print('var',var)
            if name == 'key':
                continue
            kwargs.update({name: get(var, tracker)})
#        print('function and kwargs', self.fn, kwargs)
        tracker[self] = self.fn(key, **kwargs)
        return tracker[self]






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

    def get(self, ins=dict()):
        print(len(self.parent.get(ins)), self.index, self.parent.get(ins))
        return self.parent.get(ins)[self.index]



class List(list):

    def __init__(self, _eval, shapes, dtypes, args=[], kwargs={}):

        # for convenience we only deal with kwargs, and thus transform
        # any given are into a kwarg based on the function signature
        signature = list(inspect.signature(_eval).parameters.keys())
        print('ICICICI', args, kwargs)
        for arg, name in zip(args, signature[:len(args)]):
            kwargs.update({name: arg})
        self.kwargs = kwargs
        all_tensors = list(kwargs.values())
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
            for item in self.kwargs.values():
                if hasattr(item, 'reset_value'):
                    item.reset_value(True)


    def get(self, ins=dict()):
        if self.eval_value is not None:
            return self.eval_value

        # kwarg dictionnary
        kwargs = dict()
        for name, var in self.kwargs.items():
            kwargs.update({name: get(var,ins)})

        print('HERE', self._eval, kwargs, self.kwargs)
        if 'args' in kwargs:
            args = kwargs['args']
            del kwargs['args']
        self.eval_value = self._eval(*args, **kwargs)
        return self.eval_value


class Variable(Tensor):

    def __init__(self, value, name='', trainable=True):
        self.trainable = trainable
        if not isinstance(value, jax.interpreters.xla.DeviceArray):
            self.value = np.array(value)
            if numpy.isscalar(value):
                shape = ()
                dtype = type(value)
            else:
                shape = self.value.shape
                dtype = self.value.dtype
        else:
            self.value = value
            shape = value.shape
            dtype = value.dtype
        super().__init__(None, shape=shape, dtype=dtype, name=name,
                         all_dependencies=[self])

    def __repr__(self):
        return '(Variable: ' + self.print_name + 'dtype=' + str(self.dtype) + \
               ', shape='+str(self.shape) + ', trainable='+str(self.trainable) + ')'


    def get(self, tracker={}):
        if self in tracker:
            return tracker[self]
        else:
            tracker[self] = self.value
            return tracker[self]


class Placeholder(Tensor):

    def __init__(self, shape, dtype, name=''):
        name = name
        super().__init__(None, shape=shape, dtype=dtype, name=name,
                         all_dependencies=[self])

    def __repr__(self):
        return '(Placeholder: ' + self.print_name + 'dtype=' + str(self.dtype) + \
               ', shape='+str(self.shape) + ')'

    def get(self, tracker):
        if self not in tracker:
            raise ValueError(' no value given for placeholder {}'.format(self))
        return tracker[self]
