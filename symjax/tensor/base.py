import re
from functools import wraps

import jax
import jax.numpy as jnp
import numpy

import symjax


def _add_method(cls):
    # important we keep the self inside the function call !
    def decorator(func, name=""):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        if name == "":
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
        cond1 = isinstance(item, Tensor) or type(item) in [Constant, OpTuple]
        #        cond2 = isinstance(item, jax.interpreters.partial_eval.JaxprTracer)
        cond3 = callable(item)
        return cond1 and not cond3  # (cond1 or cond2) and cond3


_numpy_signature_re = re.compile(r"^([\w., ]+=)?\s*[\w\.]+\(.*\)$")


def update_numpydoc(docstr, fun, op):
    """Transforms the numpy docstring to remove references of
       parameters that are supported by the numpy version but not the JAX version"""

    # Some numpy functions have an extra tab at the beginning of each line,
    # If this function is one of those we remove this extra tab from all the lines
    if not hasattr(op, "__code__"):
        return docstr
    if docstr[:4] == "    ":
        lines = docstr.split("\n")
        for idx, line in enumerate(lines):
            lines[idx] = line.replace("    ", "", 1)
        docstr = "\n".join(lines)

    begin_idx = docstr.find("Parameters")
    begin_idx = docstr.find("--\n", begin_idx) + 2
    end_idx = docstr.find("Returns", begin_idx)

    parameters = docstr[begin_idx:end_idx]
    param_list = parameters.replace("\n    ", "@@").split("\n")
    for idx, p in enumerate(param_list):
        param = p[: p.find(" : ")].split(", ")[0]
        if param not in op.__code__.co_varnames:
            param_list[idx] = ""
    param_list = [param for param in param_list if param != ""]
    parameters = "\n".join(param_list).replace("@@", "\n    ")
    return docstr[: begin_idx + 1] + parameters + docstr[end_idx - 2 :]


def get_output_tree(
    jax_function, *args, **kwargs,
):

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
    who_static = [int(not isvar(arg)) for arg in args]
    static_args = [arg for i, arg in zip(who_static, args) if i]
    var_args = [arg for i, arg in zip(who_static, args) if not i]

    # we need to define an abstract function that only takes as input the
    # non-static arguments, internally join them with the static ones
    # and return the output. This is because the jax shape inference
    # functions does not work with static arguments (such as the dimensions
    # of the transpose function)
    def abstract_func(*args, **kwargs):
        all_args = _args_formatting(args, static_args, who_static)
        return jax_function(*all_args, **kwargs, **static_kwargs)

    # now we evaluate the shape from the jax built-in function
    tree = jax.eval_shape(abstract_func, *var_args, **var_kwargs)
    return tree


def jax_wrap(func, insert_default_kwargs=True, doc_func=None, is_method=False):
    if doc_func is None:
        doc_func = func

    @wraps(doc_func)
    def op(*args, seed=None, **kwargs):
        # now we evaluate the shape from the jax built-in function

        # if there is a name we remove it for now to use the jax tracer
        if "name" in kwargs:
            op_name = kwargs["name"]
            del kwargs["name"]
        else:
            op_name = None

        # first we check if we are in a random function to be careful
        # with the key
        from . import random

        # this is just to get shape and dtype so we do not bother
        # to use the correct seed yet
        if func in random._RANDOM_FUNCTIONS:
            args = (jax.random.PRNGKey(0),) + args

        tree = get_output_tree(func, *args, **kwargs)

        # now we determine if it is an Op or a Tuple object based on the
        # infered shape
        if type(tree) == list or type(tree) == tuple:
            shapes = [t.shape for t in tree]
            dtypes = [t.dtype for t in tree]
            return OpTuple(
                *args,
                _jax_function=func,
                _shapes=shapes,
                _dtypes=dtypes,
                name=op_name,
                **kwargs,
            )
        elif func in random._RANDOM_FUNCTIONS:
            shape, dtype = tree.shape, tree.dtype
            return RandomOp(
                *args,
                _jax_function=func,
                _shape=shape,
                _dtype=dtype,
                _seed=seed,
                name=op_name,
                **kwargs,
            )
        else:
            shape, dtype = tree.shape, tree.dtype
            return Op(
                *args,
                _jax_function=func,
                _shape=shape,
                _dtype=dtype,
                name=op_name,
                **kwargs,
            )

    if not hasattr(func, "__doc__") or func.__doc__ is None:
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
        body = "\n\n".join(signatures + sections[i + 1 :])
        body = update_numpydoc(body, func, op)
        desc = "ADDITION"
        docstr = (
            "{summary}\n\nLAX-backend implementation of :func:`{fun}`.\n"
            "{lax_description}Original docstring below.\n\n{body}".format(
                summary=summary, lax_description=desc, fun=func.__name__, body=body,
            )
        )

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
            news = [
                n
                for n in news
                if isinstance(instance.__dict__[n], jax.interpreters.xla.DeviceArray)
            ]

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
                if att[:2] == "__" or att in method_exceptions:
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


class Tensor:

    __array_priority__ = 1000

    def __init__(self, _shape, _dtype, name=None, **kwargs):

        self._shape = tuple(_shape)
        self._dtype = _dtype

        if name is not None:
            assert "/" not in name
            self._name = name
        else:
            self._name = "unnamed"

        symjax.current_graph().add(self, **kwargs)

    @property
    def name(self):
        return self._name

    def _set_name(self, new_name):
        self._name = new_name

    def clone(self, givens):
        return symjax.current_graph().clone(self, givens)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self.shape)


class Constant:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "ConstantValue({})".format(type(self.value))

    def __str__(self):
        return self.__repr__()


class Op(Tensor):
    """an Op generates a Tensor object obtained from a function"""

    def __init__(
        self, *args, _jax_function, _shape=None, _dtype=None, name=None, **kwargs,
    ):
        self._fn = _jax_function.__name__
        # if the _shape and _dtype is not given (thus this class is
        # instanciated from anything else than the jax_wrap call)
        # then we get shape and dtype manually
        if _shape is None and _dtype is None:
            tree = get_output_tree(_jax_function, *args, **kwargs)
            _shape, _dtype = tree.shape, tree.dtype

        assert _shape is not None and _dtype is not None

        if name is None:
            name = _jax_function.__name__

        super().__init__(
            _shape,
            _dtype,
            name=name,
            args=args,
            kwargs=kwargs,
            jax_function=_jax_function,
            **kwargs,
        )

    def __repr__(self):

        name = "Op(name={}, fn={}, shape={}, dtype={}, scope={})"
        return name.format(self.name, self._fn, self.shape, self.dtype, self.scope)

    def __str__(self):

        return self.__repr__()


class OpTuple:
    def __init__(self, *args, _jax_function, _shapes, _dtypes, name=None, **kwargs):

        if name is None:
            name = _jax_function.__name__
        self._name = name
        symjax.current_graph().add(
            self, jax_function=_jax_function, args=args, kwargs=kwargs, **kwargs,
        )
        for i, (shape, dtype) in enumerate(zip(_shapes, _dtypes)):
            OpTupleItem(
                shape, dtype, index=i, parent=self, name=name + "[{}]".format(i),
            )

    def __repr__(self):
        successors = symjax.current_graph().successors(self)

        return "(" + ", ".join([str(a) for a in successors]) + ")"

    @property
    def name(self):
        return self._name

    def _set_name(self, name):
        self._name = name

    def __str__(self):

        return self.__repr__()

    def __iter__(self):
        """ Returns the Iterator object """
        return symjax.current_graph().successors(self)

    def __getitem__(self, item):
        assert type(item) == int
        for node in symjax.current_graph().successors(self):
            if symjax.current_graph()[self][node]["index"] == item:
                return node


class OpTupleItem(Tensor):
    def __init__(self, shape, dtype, index, parent, name):
        super().__init__(shape, dtype, name=name, parent=parent, index=index)

    def __repr__(self):

        name = "OpTupleItem(name={}, shape={}, dtype={}, scope={})"
        return name.format(self.name, self.shape, self.dtype, self.scope)

    def __str__(self):

        return self.__repr__()


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

    """

    def __init__(self, *args, _jax_function, _shape, _dtype, _seed, name, **kwargs):
        self._shape = _shape
        self._dtype = _dtype
        self._fn = _jax_function.__name__
        super().__init__(
            *args,
            _jax_function=_jax_function,
            _shape=_shape,
            _dtype=_dtype,
            name=name,
            _seed=_seed,
            **kwargs,
        )

    def __repr__(self):
        name = "RandomTensor(name={}, fn={}, shape={}, dtype={}, scope={})"
        return name.format(self.name, self._fn, self.shape, self.dtype, self.scope)


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

    def __init__(
        self, initializer, name="unnamed_variable", trainable=True, dtype=None
    ):

        self.trainable = trainable
        self.initializer = initializer
        self._dtype = dtype

        super().__init__(self.value.shape, str(self.value.dtype), name=name)

    def reset(self):
        """reset the value of the variable based on the initial one, whether
        it was an array or initializer. If it was a random initializer,
        nothing guarantees that the reset will give back the original value
        as opposed to the array case
        """
        if isinstance(self.initializer, Tensor):
            self._value = symjax.current_graph().get(self.initializer)
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
        if not hasattr(self, "_value"):
            self.reset()
        return self._value

    def update(self, update_value):
        """assign a new value for the variable"""

        self._value = symjax.current_graph().get(update_value)

    def __repr__(self):
        name = "Variable(name={}, shape={}, dtype={}, trainable={}, scope={})"
        return name.format(
            self.name, self.shape, self.dtype, self.trainable, self.scope
        )


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

    def __init__(self, shape, dtype, name="unnamed"):
        super().__init__(shape, dtype, name=name)

    def __repr__(self):
        name = "Placeholder(name={}, shape={}, dtype={}, scope={})"
        return name.format(self.name, self.shape, self.dtype, self.scope)


def placeholder_like(item, name=""):
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
        outputs = symjax.current_graph().get(symjax_outputs, feed_dict)
        return outputs

    return newfn


def clone(tensor, givens):
    return tensor.clone(givens)


def get(tensor, tracker=None):
    return symjax.current_graph().get(tensor, tracker)
