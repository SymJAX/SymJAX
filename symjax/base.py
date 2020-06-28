#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base of symjax."""

import fnmatch
import warnings

import jax
import numpy
from jax import jacfwd, jacrev

import symjax
import collections
from symjax import tensor as t
from symjax.tensor import random
import networkx as nx


def current_scope():
    """Current scope."""
    return current_graph()._scopes[-1]


def current_graph():
    """Current graph."""
    assert len(symjax._graphs)
    return symjax._graphs[-1]


class Graph(nx.DiGraph):
    def __init__(self, name, *args, **kwargs):
        self._current_scope = "/"
        self._scopes = [Scope("", graph=self)]
        self._variables = {}
        self._placeholders = {}
        self._updates = {}
        self._ops = {}
        super().__init__(*args, name=name, **kwargs)

    def add(self, tensor, branch=None, **kwargs):

        if type(tensor) == dict:
            self._updates.update(tensor)

        if not t.isvar(tensor):
            const_node = t.Constant(tensor)
            self.add_node(const_node, root=False)
            return const_node

        if tensor in self.nodes:
            return tensor

        self.scope.add(tensor)

        if isinstance(tensor, t.Placeholder) or isinstance(tensor, t.Variable):
            self.add_node(tensor, branch=branch, root=True)

        elif isinstance(tensor, t.RandomOp):
            self.add_node(
                tensor,
                branch=branch,
                root=True,
                jax_function=kwargs["jax_function"],
                seed=kwargs["_seed"],
            )

        elif type(tensor) == t.OpTupleItem:
            self.add_node(tensor, root=False)
            self.add_edge(kwargs["parent"], tensor, index=kwargs["index"])

        elif isinstance(tensor, t.Op) or isinstance(tensor, t.OpTuple):

            self.add_node(
                tensor, branch=branch, root=False, jax_function=kwargs["jax_function"],
            )

            for i, arg in enumerate(kwargs["args"]):

                # all parents should already be in, however if one
                # of the arg is like a tuple then this is a new node
                # and thus it is not already in thus we add it
                node = self.add(arg)
                if self.has_edge(node, tensor):
                    self[node][tensor]["name"] += "+arg{:02d}".format(i)
                else:
                    self.add_edge(node, tensor, name="arg{:02d}".format(i))

            for key, arg in kwargs["kwargs"].items():

                node = self.add(arg)
                if self.has_edge(node, tensor):
                    self[node][tensor]["name"] += "+" + key
                else:
                    self.add_edge(node, tensor, name=key)

        elif type(tensor) == tuple:
            self.add_node(tensor, root=False)
            for i in tensor:
                other = self.add(i)
                self.add_edge(other, tensor)

        return tensor

    def roots(self, nodes, roots=None):

        if roots is None:
            roots = []

        if type(nodes) == tuple or type(nodes) == list:
            for node in nodes:
                self.roots(node, roots)
        else:
            for i in nx.algorithms.ancestors(self, nodes):
                if self.nodes[i]["root"]:
                    roots.append(i)

        return list(set(roots))

    def clone(self, node, givens):
        names = [
            "{}{}->{}{}".format(m.scope, m.name, v.scope, v.name)
            for m, v in givens.items()
        ]
        names.sort()
        name = "_".join(names)
        branch = givens.copy()
        for start in givens.keys():
            for path in nx.all_simple_paths(self, source=start, target=node):
                for n in path[1:]:
                    if n in branch:
                        continue
                    args, kwargs = self._get_args_kwargs(n, evaluate=False)
                    args = [branch[arg] if arg in branch else arg for arg in args]
                    for arg in kwargs.keys():
                        if arg in branch:
                            kwargs[arg] = branch[arg]
                    fun = self.get_node_attribute(n, "jax_function")
                    kwargs.update(
                        {
                            "_jax_function": fun,
                            "_shape": n._shape,
                            "_dtype": n._dtype,
                            "name": n.name + "_clone",
                        }
                    )
                    branch[n] = t.Op(*args, **kwargs)
        return branch[node]

    def get_node_attribute(self, node, attr):
        return nx.get_node_attributes(self, attr)[node]

    def get(self, item, tracker=None, seed=None):
        """
        Example:

            >>> import symjax.tensor as T
            >>> w = T.ones(10).sum()+4
            >>> v = T.Variable(1., name='v', dtype='float32')
            >>> T.get(w+v)
            DeviceArray(15., dtype=float32)
        """

        if tracker is None:
            tracker = {}

        if not t.isvar(item):
            return item

        if type(item) == list:
            return [self.get(i, tracker) for i in item]
        elif type(item) == tuple:
            return tuple([self.get(i, tracker) for i in item])

        elif item in tracker:
            return tracker[item]

        elif isinstance(item, t.Placeholder):
            if item not in tracker:
                raise ValueError(" no value given for placeholder {}".format(item))

        elif isinstance(item, t.Variable) or type(item) == t.Constant:
            tracker[item] = item.value
            return tracker[item]

        elif type(item) == t.OpTupleItem:

            preds = list(self.predecessors(item))
            assert len(preds) == 1
            parent = preds[0]
            index = self.get_edge_data(parent, item)["index"]
            return self.get(parent, tracker)[index]

        elif isinstance(item, t.RandomOp):
            seed = seed or numpy.random.randint(0, 1000000)
            intra_seed = 1 or self.get_node_attribute(item, "seed")
            key = jax.random.PRNGKey(seed + intra_seed)
            args, kwargs = self._get_args_kwargs(item, tracker)
            tracker[item] = self.get_node_attribute(item, "jax_function")(
                key, *args, **kwargs
            )
            return tracker[item]

        elif isinstance(item, t.Op) or isinstance(item, t.OpTuple):

            # first get the actual parents nodes (aka inputs to the function)
            args, kwargs = self._get_args_kwargs(item, tracker)
            tracker[item] = self.get_node_attribute(item, "jax_function")(
                *args, **kwargs
            )

            return tracker[item]

        else:
            return item

    def _get_args_kwargs(self, node, tracker=None, evaluate=True):

        if evaluate:
            assert tracker is not None

            all_args = {
                self.get_edge_data(parent, node)["name"]: self.get(parent, tracker)
                for parent in self.predecessors(node)
            }
        else:
            all_args = {
                self.get_edge_data(parent, node)["name"]: parent
                for parent in self.predecessors(node)
            }
        # now we inspect if there are duplicate args
        for arg in list(all_args.keys()):
            if "+" in arg:
                items = arg.split("+")
                for item in items:
                    all_args.update({item: all_args[arg]})
                del all_args[arg]

        args_names = [name for name in all_args.keys() if "arg" == name[:3]]
        args_names.sort()

        args = [all_args[name] for name in args_names]
        for name in args_names:
            del all_args[name]

        return args, all_args

    @property
    def scope(self):
        return self._scopes[-1]

    def variables(self, trainable=True):
        variables = [n for n in self.nodes if isinstance(n, t.Variable)]
        if trainable is None:
            return variables
        return [v for v in variables if v.trainable == trainable]

    @property
    def placeholders(self):
        placeholders = [n for n in self.nodes if type(n) == t.Placeholder]
        return placeholders

    @property
    def ops(self):
        ops = [n for n in self.nodes if type(n) in [t.Op, t.OpTupleItem]]
        return ops

    def updates(self, name="*"):
        sub = {}
        for v in self._updates:
            matched = fnmatch.filter([v.name], name)
            if len(matched):
                sub[v] = self._updates[v]
        return sub


class Scope:
    """
    Defining scope for any variable/operation to be in.

    Example
    -------

    .. doctest::
        >>> import symjax
        >>> import symjax.tensor as T
        >>> v = T.Variable(1, name='v')
        >>> # the current (default) scope is the root of the graph
        >>> print(v.scope)
        /
        >>> with symjax.Scope('my_scope'):
        ...     w = T.Variable(2, name='w')
        ...     out = v*w
        >>> print(out.scope)
        /my_scope/
        >>> print(w.scope)
        /my_scope/
        >>> #it is also possible to keep a scope persistently
        >>> scope = symjax.Scope('second_scope')
        >>> with scope:
        ...     other = out * w
        >>> print(other.scope)
        /second_scope/
        >>> # this allows to keep track directly of internal ops
        >>> print(scope.ops)
        [Op(name=multiply, fn=multiply, shape=(), dtype=int32, scope=/second_scope/)]


    """

    def __init__(self, name, graph=None, seed=None):
        """Constructor."""
        if graph is None:
            graph = current_graph()
        self.graph = graph
        self.name = name
        self.full_name = None

    def __enter__(self):
        """Set global variables."""
        if len(self.name):
            self.graph._current_scope += self.name + "/"
        self.full_name = self.graph._current_scope
        self.graph._scopes.append(self)

    def __exit__(self, *a):
        """Delete globals."""
        self.graph._scopes.pop(-1)
        if self.graph._scopes[-1].full_name is not None:
            self.graph._current_scope = self.graph._scopes[-1].full_name
        else:
            self.graph._current_scope = "/"

    def save_variables(self, path):
        """Save graph."""
        numpy.savez(
            path, **dict([(v.name, symjax.tensor.get(v)) for v in self.variables])
        )

    @property
    def variables(self):
        return get_variables(self.full_name + "*")

    @property
    def ops(self):
        return get_ops(self.full_name + "*")

    @property
    def placeholders(self):
        return get_placeholders(self.full_name + "*")

    def load_variables(self, path):
        """Load graph."""

        data = numpy.load(path)
        for name, value in data.items():
            self.variable[name].update(value)

    def reset(self):

        for var in self.variables:
            var.reset()

    def add(self, tensor):

        if not t.isvar(tensor) or type(tensor) == tuple:
            return

        # fake graph entrance if not used by user
        if self.full_name is None:
            self.__enter__()

        # in this case we were given updates
        if type(tensor) == dict:
            self.graph._updates.update(tensor)
            return

        tensor.scope = self.full_name
        name = self.full_name + tensor.name
        if isinstance(tensor, symjax.tensor.Placeholder):
            names = self.graph._placeholders
        elif isinstance(tensor, symjax.tensor.Variable):
            names = self.graph._variables
        else:
            names = self.graph._ops

        if name not in names.keys():
            names[name] = tensor
            return

        count = 1
        while True:
            if name + "_" + str(count) in names.keys():
                count += 1
            else:
                break
        names[name + "_" + str(count)] = tensor
        tensor._set_name(tensor.name + "_" + str(count))


def reset_variables(name="*", trainable=None):
    """
    utility to reset variables based on their names

    Parameters
    ----------

    name: str (default=*)
        the name (or part of the name) of all the variables that should be
        reset, it can include the glob (*) searching for all matching
        names

    trainable: bool or None (optional, default=None)
        is not None, it will only reset from the matched variables the ones that
        trainable attribute matches the given one


    Returns
    -------

    None

    Example
    -------

    .. doctest::

        >>> import symjax
        >>> w = symjax.tensor.Variable(1., name='w', dtype='float32')
        >>> x = symjax.tensor.Variable(2., name='x', dtype='float32')
        >>> f = symjax.function(outputs=[w, x], updates={w:w + 1,x:x + 1})
        >>> for i in range(10):
        ...    print(f())
        [array(1., dtype=float32), array(2., dtype=float32)]
        [array(2., dtype=float32), array(3., dtype=float32)]
        [array(3., dtype=float32), array(4., dtype=float32)]
        [array(4., dtype=float32), array(5., dtype=float32)]
        [array(5., dtype=float32), array(6., dtype=float32)]
        [array(6., dtype=float32), array(7., dtype=float32)]
        [array(7., dtype=float32), array(8., dtype=float32)]
        [array(8., dtype=float32), array(9., dtype=float32)]
        [array(9., dtype=float32), array(10., dtype=float32)]
        [array(10., dtype=float32), array(11., dtype=float32)]
        >>> # reset only the w variable
        >>> symjax.reset_variables('w')
        >>> # reset all variables
        >>> symjax.reset_variables('*')

    """

    variables = get_variables(name, trainable)
    for var in variables:
        var.reset()


def save_variables(name, path):
    """Save graph."""
    matched = fnmatch.filter(symjax._variables.keys(), name)
    numpy.savez(
        path,
        **dict(
            [
                (
                    symjax._variables[v].scope + symjax._variables[v].name,
                    symjax.tensor.get(symjax._variables[v]),
                )
                for v in matched
            ]
        )
    )


def load_variables(name, path_or_file, scope_mapping=None):
    """Load graph."""

    if type(path_or_file) == str:
        if path_or_file[-4:] != ".npz":
            path_or_file += ".npz"

    scope_mapping = scope_mapping or {}

    matched = fnmatch.filter(symjax._variables.keys(), name)
    data = numpy.load(path_or_file)
    for name in matched:
        if symjax._variables[name].scope in scope_mapping:
            name_in_file = (
                scope_mapping[symjax._variables[name].scope]
                + "/"
                + symjax._variables[name].name
            )
        else:
            name_in_file = name
        if name_in_file not in data:
            raise Warning("{} not in loaded file".format(name_in_file))
        symjax._variables[name].update(data[name_in_file])


def get_variables(name="*", trainable=None):
    matched = current_graph().variables(trainable)
    output = []
    for m in matched:
        if len(fnmatch.filter([m.name], name)):
            output.append(m)
    return output


def get_placeholders(name="*"):
    """
    Same as symjax.variable but for placeholders
    """
    matched = current_graph().placeholders
    output = []
    for m in matched:
        if len(fnmatch.filter([m.name], name)):
            output.append(m)
    return output


def get_ops(name="*"):
    """
    Same as symjax.variable but for ops
    """
    matched = current_graph().ops
    output = []
    for m in matched:
        if len(fnmatch.filter([m.scope + m.name], name)):
            output.append(m)
    return output


def add_updates(udpates):
    current_scope().add_updates(updates)


def get_updates(name="*"):
    """
    Same as symjax.get_variables but for updates
    """
    return current_graph().updates(name)


def gradients(scalar, variables):
    """Compute the gradients of a scalar w.r.t to a given list of variables.

    Arguments
    ---------
    scalar: :class:`symjax.tensor.base.Tensor`
        the variable to differentiate

    variables: List or Tuple
        the variables used to compute the derivative.

    Returns
    -------

        gradients: Tuple
            the sequency of gradients ordered as given in the input variables

    Example
    -------

    .. doctest::

        >>> import symjax
        >>> w = symjax.tensor.ones(3)
        >>> x = symjax.tensor.Variable(2., name='x', dtype='float32')
        >>> l = (w ** 2).sum() * x
        >>> g = symjax.gradients(l, [w])[0]
        >>> f = symjax.function(outputs=g, updates={x:x + 1})
        >>> for i in range(2):
        ...    print(f())
        [4. 4. 4.]
        [6. 6. 6.]

    """
    if numpy.prod(scalar.shape) != 1:
        raise RuntimeError("the variable to differentiate is not a scalar")
    if not isinstance(scalar, t.Tensor):
        raise RuntimeError("the variable used in gradients should be a Tensor type")

    if scalar.shape != ():
        scalar = scalar.sum()
    if isinstance(variables, t.Tensor):
        input_variables = [variables]
        input_list = False
    else:
        input_variables = variables
        input_list = True

    # get all the roots of the scalar, this is needed as otherwise they are not
    # as the input of the gradient function and thus a change of
    # their value will not change the gradient computation, we also ensure
    # uniqueness
    all_roots = list(set(current_graph().roots(scalar) + input_variables))

    # get the argnum of the variables that we differentiate one
    argnums = [all_roots.index(var) for var in input_variables]

    # create a dummy function that is needed for jax to compute a gradient func
    # this function is the one that builds the graph of computation from all
    # roots
    # to the scalar varible s.t. automatic diffenrentiation can be applied
    def fn(*args):
        return current_graph().get(scalar, dict(zip(all_roots, list(args))))

    # now we obtain the grad function. In fact, Jax returns a function that,
    # when it is called, returns the gradient values, this function is then
    # used to generate the Tuple of symbolic variables
    grad_fn = jax.grad(fn, argnums)
    wrap_fn = t.jax_wrap(grad_fn, False)
    if input_list:
        return wrap_fn(*all_roots)
    else:
        return wrap_fn(*all_roots)[0]


def jacobians(tensor, variables, mode="forward"):
    """Compute the jacobians of a tensor w.r.t to a given list of variables.

    The tensor needs not to be a vector, but will be treated as such. For
    example if tensor.shape is (10, 3, 3) and a variable shape if (10, 10)
    the resulting jacobian has shape (10, 3, 3, 10, 10). It is possible
    to specify the mode forward or backward. For tall jacobians, forward
    is faster and vice-versa.

    Arguments
    ---------

        vector: Tensor
            the variable to differentiate

        variables: List or Tuple
            the variables used to compute the derivative.

    Returns
    -------

        jacobians: Tuple
            the sequency of gradients ordered as given in the input variables
            :param tensor:
            :param mode:
    """
    # get all the roots of the scalar, this is needed as otherwise they are not
    # as the input of the gradient function and thus a change of
    # their value will not change the gradient computation, we also ensure
    # uniqueness
    all_roots = list(set(tensor.roots + variables))

    # get the argnum of the variables that we differentiate one
    argnums = [all_roots.index(var) for var in variables]

    # create a dummy function that is needed for jax to compute a gradient func
    # this function is the one that builds the graph of computation from
    # all roots
    # to the scalar varible s.t. automatic diffenrentiation can be applied
    def fn(*args):
        return symjax.tensor.get(tensor, dict(zip(all_roots, list(args))))

    # now we obtain the jacobian function. In fact, Jax returns a function that
    # when it is called, returns the jacobian values, this function is then
    # used to generate the Tuple of symbolic variables
    if mode == "forward":
        jacob_fn = jacfwd(fn, argnums)
    elif mode == "backward":
        jacob_fn = jacrev(fn, argnums)
    else:
        raise RuntimeError(
            "mode {} not recognized, use forward or backward".format(mode)
        )
    wrap_fn = t.jax_wrap(jacob_fn, False)
    return wrap_fn(*all_roots)


class function:
    """Generate a user function that compiles a computational graph.

    Based on given inputs, outputs and update policy of variables. This
    function internally jit compile the underlying jax computational
    graph for performances.

    Arguments
    ---------

        classargs: trailing tuple
            the inputs to the function to be compiled. The tuple should
            contain all the placeholders that are roots of any output
            given of the function and update values

        outputs: List (optional)
            the outputs of the function, if a single element, it can be
            given as a standalone and not a list

        updates: Dict (optional)
            the dictionnary of updates as per {var:new_value} for any
            variable of the graph

        backend: 'cpu' or 'gpu'
            the backend to use to run the function on

        default_value: not implemented
            not implemented

    Returns
    -------

        callable:
            the user frontend function that takes the specified inputs,
            returns the specified outputs and perform internally the
            updates

    Examples
    --------

        >>> import symjax
        >>> import symjax.tensor as T
        >>> x = T.ones((4, 4))
        >>> xs = x.sum() + 1
        >>> f = symjax.function(outputs=xs)
        >>> print(f())
        17.0

        >>> w = T.Variable(0., name='w', dtype='float32')
        >>> increment = symjax.function(updates={w: w + 1})
        >>> for i in range(10):
        ...     increment()
        >>> print(w.value)
        10.0

    """

    def __init__(
        self,
        *classargs,
        outputs=[],
        updates=None,  # noqa
        device=None,
        backend=None,
        default_value=None
    ):
        """Initialize."""
        # check the given updates (if any) and ensure that they only
        # update Variable objects
        if updates is None:
            updates = {}

        for update in updates.keys():
            if not isinstance(update, t.Variable):
                raise RuntimeError(
                    "{} is not a Variable and cannot be updated".format(update)
                )

        # ensure that all inputs are actual placeholders or variables
        for arg in classargs:
            if not isinstance(arg, t.Tensor):
                raise RuntimeError(
                    "{} is not a Tensor type. Only tensor types can be"
                    + "function inputs".format(arg)
                )

        # gather all roots, they need to be explicit as inputs of the
        # underlying functions otherwise they are treated as constants
        # and any change in their value will not appear when running the
        # function
        outs = list(updates.values())
        outs += [outputs] if isinstance(outputs, t.Tensor) else outputs
        self.all_roots = set(symjax.current_graph().roots(outs))
        self.classargs = classargs
        self.outputs = outputs
        items = list(updates.items())
        self.updates_keys = [item[0] for item in items]
        self.updates_values = [item[1] for item in items]
        for i in range(len(items)):
            if self.updates_keys[i].shape != self.updates_values[i].shape:
                warnings.warn(
                    "Variable and update {} {}".format(
                        self.updates_keys[i], self.updates_values[i]
                    )
                    + "are not the same shape... attempting to reshape"
                )
                self.updates_values[i] = t.reshape(
                    self.updates_values[i], self.updates_keys[i].shape
                )
            if self.updates_keys[i].dtype != self.updates_values[i].dtype:
                warnings.warn(
                    "Variable and update {} {}".format(
                        self.updates_keys[i], self.updates_values[i]
                    )
                    + "are not the same dtype... attempting to cast"
                )
                self.updates_values[i] = t.cast(
                    self.updates_values[i], self.updates_keys[i].dtype
                )

        # check the function inputs, they must be at least contain all the
        # placeholders needed to compute the outputs values
        placeholders_in_root = filter(
            lambda x: isinstance(x, t.Placeholder), self.all_roots
        )

        # check for
        non_givens = set(placeholders_in_root) - set(self.classargs)
        if len(non_givens) > 0:
            raise RuntimeError(
                "Missing placeholders form the function inputs: {}".format(non_givens)
            )

        # the roots are made of variables, random tensors, placeholders. We
        # already ensured that all placeholders are given as inputs to the
        # function. Now we must ensure that the other ones will also be given
        # as inputs to not be treated as constants by jax.
        # we also remove update keys because we will expicitly feed them
        self.extra_inputs = set(self.all_roots) - (
            set(self.classargs).union(self.updates_keys)
        )
        self.extra_inputs = list(self.extra_inputs)
        allargs = list(self.classargs) + self.updates_keys + self.extra_inputs

        def to_jit(*jitargs, seed):

            feed_dict = dict(zip(allargs, jitargs))
            feed_dict.update({"rng": seed})
            outputs = [self.outputs, self.updates_values]
            return symjax.current_graph().get(outputs, feed_dict)

        # take care of the presence of -1 dimensions
        to_vectorize = [0 if 0 in a.shape else None for a in allargs]

        if any(to_vectorize):
            self.jited = jax.jit(
                jax.vmap(to_jit, to_vectorize), device=device, backend=backend
            )
        else:
            # we compile our underlying function using jit for performances
            self.jited = jax.jit(to_jit, device=device, backend=backend)

        # define the frontend function that takes as input the inputs variables
        # and internally compute and update the variables from updates if any
        def meta(*fnargs, rng):

            # ensure that the number of arguments is correct
            assert len(fnargs) == len(self.classargs)
            for fnarg, classarg in zip(fnargs, self.classargs):
                if hasattr(fnarg, "shape"):
                    if fnarg.shape != classarg.shape and 0 not in classarg.shape:
                        raise RuntimeError(
                            "wrong input given for {}".format(classarg)
                            + ", given is {}".format(fnarg)
                            + ", shape={}".format(fnarg.shape)
                        )

            # retreive the function outputs, updated values and apply them
            jited_add_inputs = symjax.current_graph().get(
                self.updates_keys + self.extra_inputs, tracker={"rng": rng}
            )
            jitoutputs, jitupdates = self.jited(*fnargs, *jited_add_inputs, seed=rng)
            for key, update in zip(self.updates_keys, jitupdates):
                key.update(update)
            if isinstance(jitoutputs, jax.interpreters.xla.DeviceArray):
                return jax.api.device_get(jitoutputs)
            else:
                npy_jitoutputs = [
                    jax.api.device_get(arr)
                    if isinstance(arr, jax.interpreters.xla.DeviceArray)
                    else arr
                    for arr in jitoutputs
                ]
                return npy_jitoutputs

        self.meta = meta

    def __call__(self, *args, rng=None):
        """Callable fn."""
        # in the presence of RandomTensor(s) in the graph, we keep track of the
        # number of functions calls to keep accumulating the PRNGKey of the jax
        # key, otherwise each function call returns the same realisation

        if rng is None:
            rng = random._seed
            random._seed += 1
        pargs = [numpy.array(arg) if type(arg) == list else arg for arg in args]
        outputs = self.meta(*pargs, rng=rng)
        if type(outputs) == list:
            if len(outputs) == 0:
                return None
        return outputs
