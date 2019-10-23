import jax
import jax.numpy as np
from jax.random import _is_prng_key
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
    # we need to take care of slices first as they are
    # not hashable
#    if type(item) == slice:
#        return item
    # important to do this case first to deal with a list of
    # unashable variables
    if type(item) == list or type(item) == tuple:
        current = [get(i, tracker) for i in item]
        return current
    elif item in tracker:
        return tracker[item]
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
#        print(item, (cond1 or cond2) and cond3)
        return (cond1 or cond2) and cond3


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

    def __init__(self, fn, name='', docstring=None, is_root=False):
        """function that produces a new Op based on a given function"""
        self.fn = fn
        self.name = name

    def __call__(self, *args, _shape=None, _dtype=None, name='', **kwargs):
        name = 'Op(' + self.name + ')' if name == '' else\
                    'Op(' + self.name + ', ' + name + ')'
        return Tensor(self.fn, args=args, kwargs=kwargs,
                      name='Op(' + self.name + ')')


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

    def __init__(self, fn_or_tensor, args=[], kwargs={}, name='', roots=[],
                 is_root=False):
        print(args, kwargs)
        self.kwargs = kwargs
        self.fn = fn_or_tensor
        self.is_fn = callable(fn_or_tensor)
        self.name = name
        self.print_name = 'name=' + name + ', ' if name != '' else ''

        if self.is_fn:
            # for convenience we only deal with kwargs, and thus transform
            # any given arg into a kwarg based on the function signature
            signature = list(inspect.signature(fn_or_tensor).parameters.keys())
            for arg, name in zip(args, signature[:len(args)]):
                self.kwargs.update({name: arg})
            kwargs = self.kwargs.copy()

            # set roots
            self.roots = list() if not is_root else [self]
            for value in kwargs.values():
                if hasattr(value, 'roots'):
                    self.roots += value.roots
            self.roots = list(set(self.roots))

            # set shape and dtype
            # if we have to use the automatic shape evaluation then we have to
            # take care of some problematic cases happening when a parameter
            # such as the shape in the reshape function is given. In fact, when
            # trying to infer the shape, each input ( arg or kwarg) is given as
            # a tracer to monitor internal computations. But those parameters
            # are not the ones that support such assignments and we thus have
            # to infer the shape using a slightly tweaked function

            # get the function signature (inputs and keywords)
            signature = list(inspect.signature(self.fn).parameters.keys())

            # the kwargs that are constant are removed and put into
            # extra_kwargs as they should not be probed to infer shape and
            #dtype
            extra_kwargs = {}
            for name, arg in list(kwargs.items()):
                if name == 'key' or not isvar(arg):
                    extra_kwargs.update({name: arg})
                    del kwargs[name]
            # now use the builin function to infer shape and dtype given a
            # lambda jax function
            self.kwargs, self.extra_kwargs = kwargs, extra_kwargs
            tree = jax.eval_shape(lambda **b: self.fn(**b, **extra_kwargs),
                                  **kwargs)
            self.shape, self.dtype = tree.shape, tree.dtype
        else:
            # set up the roots
            self.roots = roots

            # set shape and dtype
            if type(self.fn) is tuple:
                self.shape = self.fn[0]
                self.dtype = self.fn[1]
            else:
                self.shape = self.fn.shape
                self.dtype = self.fn.dtype



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
        for name, var in list(self.kwargs.items()):
            kwargs.update({name: get(var, tracker)})
        tracker[self] = self.fn(**kwargs, **self.extra_kwargs)
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
        super().__init__(_eval, args=(key,)+args, kwargs=kwargs, name=name,
                         is_root=True)

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
        for name, var in self.kwargs.items():
            if name == 'key':
                continue
            kwargs.update({name: get(var, tracker)})
        tracker[self] = self.fn(key, **kwargs)
        return tracker[self]



class SubTensor(Tensor):

    def __init__(self, tensor, index, parent, roots, name=''):
        self.parent = parent
        self.index = index
        super().__init__(tensor, roots=roots, name=name)

    def get(self, tracker=dict()):
        return self.parent.get(tracker)[self.index]



class List:

    def __init__(self, fn_or_list, shapes=None, dtypes=None, args=[], kwargs={}):

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

        self.items = [SubTensor((shape, dtype), i, self, roots=[])
                 for (i, shape), dtype in zip(enumerate(shapes), dtypes)]
    def __iter__(self):
        self.current_i = 0
        return self

    def __next__(self):
        self.current_i += 1
        if self.current_i > len(self):
            return StopIteration
        return self[self.current_i-1]

    def __getitem__(self, key):
            return self.items[key]

    def __len__(self):
            return len(self.items)

    def get(self, tracker=dict()):
        if self in tracker:
            return tracker[self]

        # if it was just a list of Tensor then evaluate them
#        if not self.is_fn:
#            values = list()
#            for var in self.items:
#                values.append(get(var, tracker))
#            tracker[self] = values
#            return values

        # kwarg dictionnary
        args = list()
        for var in self.args:
            args.append(get(var, tracker))

        kwargs = dict()
        for name, var in self.kwargs.items():
            kwargs.update({name: get(var, tracker)})

        # we add the list object itself into the dictionnary first
        tracker[self] = self.fn(*args, **kwargs)

        # then we add each element of the list into the dictionnary
        for i in range(len(self)):
            tracker[self[i]] = tracker[self][i]

        return tracker[self]


class Slice(Tensor):
    def __init__(self, islice):
        self.islice = islice
        self.print_name = 'Slice'

    def __repr__(self):
        return 'SLICE TO CHANGE'

    def get(self, tracker={}):
        if self in tracker:
            return tracker[self]
        if type(islice.start) != int:
            start = islice.start.get(tracker)
        else:
            start = islice.start
        #
        if type(islice.stop) != int:
            end = islice.stop.get(tracker)
        else:
            end = islice.stop
        #
        if type(islice.step) != int:
            step = islice.step.get(tracker)
        else:
            step = islice.step
        tracker[self] = slice(start, end, step)
        return tracker[self]

class Variable(Tensor):

    def __init__(self, value, name='', trainable=True):
        self.trainable = trainable
        if not isinstance(value, jax.interpreters.xla.DeviceArray):
            self.value = np.asarray(value)
            self.init_value = self.value + 0
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
        super().__init__(self.value, name=name, roots=[self])

    def reset(self):
        self.value = self.init_value

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
        super().__init__((shape, dtype), name=name, roots=[self])

    def __repr__(self):
        return '(Placeholder: ' + self.print_name + 'dtype=' + str(self.dtype) + \
               ', shape='+str(self.shape) + ')'

    def get(self, tracker):
        if self not in tracker:
            raise ValueError(' no value given for placeholder {}'.format(self))
        return tracker[self]

def placeholder_like(item, name=''):
    return Placeholder(item.shape, item.dtype, name=name)

# we transform into
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
    print(feed_dict)
    return output.get(dict(feed_dict))

