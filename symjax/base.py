#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base of symjax."""

import fnmatch

import jax
import numpy
from jax import jacfwd, jacrev

import symjax
from symjax import tensor as t
from symjax.tensor import random
import networkx as nx
import re
import collections


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def current_scope():
    """Current scope."""
    return current_graph()._scopes[-1]


def current_graph():
    """Current graph."""
    assert len(symjax._graphs)
    return symjax._graphs[-1]


class Graph(nx.DiGraph):
    def __init__(self, name, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._name = name
        self.reset()
        self._jax_to_op = {}

    def reset(self):
        self.clear()
        self._current_scope = "/"
        self._scopes = [Scope("", graph=self)]
        self._updates = {}
        self._scopes_history = []
        self._branches = {}

    def __repr__(self):
        msg = "Graph(name:{}, n_edges:{}, n_nodes:{})".format(
            self.name, self.n_edges, self.n_nodes
        )
        return msg

    def __str__(self):
        return self.__repr__()

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def name(self):
        return self._name

    def add_updates(self, updates):
        # TODO
        # we add just in case there are some lists or something
        # this might not be needed

        for i in updates.values():
            self._add(i)
        self._updates.update(updates)

    def _get_name_scope(self, name, tensor):
        return self.scope._get_name_scope(name, tensor)

    def is_connected(self, node_1, node_2, directed=True):
        """check if two nodes are connected in the graph.

        This function is useful to check wheter two nodes (or Op)
        as connected in the graph and are thus dependent on each other.
        For example this is useful to select on trainable variables that
        affect a specific tensor.

        Args:

            node_1: Tensor

            node_2: Tensor

            directed: bool

                whether to test for both directions or not, if the graph
                is not directed then this parameter has no effect.
                If ``False`` then the function will return ``True``
                is the nodes are connected no matter the direction

        Returns:

        bool

        """
        if directed:
            return nx.has_path(self, node_1, node_2)
        else:
            return nx.has_path(self, node_1, node_2) or nx.has_path(
                self, node_2, node_1
            )

    def _add(self, tensor, *args, _attrs=None, **kwargs):

        _attrs = _attrs or {}

        # first we check is the not is a hashable, if it is not
        # already in the graph
        if isinstance({}, collections.Hashable):
            if tensor in self.nodes:
                return tensor

        # then, if the node is a constant (not tensor, variable, ...)
        elif not t.isvar(tensor):

            # if it is hashable then we can add it as is as a node
            # and we return it since it is the same object
            if isinstance({}, collections.Hashable):
                self.add_node(tensor, root=False)
                return tensor
            # otherwise we have to make it hashable, to do so we use
            # the constant object
            # during creation of the object, it will be added to the
            # graph automatically, we can return the object
            return t.Constant(tensor)

        # now check if it is a list of a tuple
        elif type(tensor) == list or type(tensor) == tuple:
            return t.Tuple(*tensor)

        if isinstance(tensor, t.Tensor):
            self.add_node(tensor, **_attrs)

            if isinstance(tensor, t.Op):

                for i, arg in enumerate(args):

                    node = self._add(arg)
                    if self.has_edge(node, tensor):
                        self[node][tensor]["name"] += "+arg" + str(i)
                    else:
                        self.add_edge(node, tensor, name="arg" + str(i))

                for key, arg in kwargs.items():

                    node = self._add(arg)
                    if self.has_edge(node, tensor):
                        self[node][tensor]["name"] += "+" + key
                    else:
                        self.add_edge(node, tensor, name=key)

        return tensor

    def roots(self, nodes, roots=None):

        if roots is None:
            roots = []

        if type(nodes) == tuple or type(nodes) == list:
            for node in nodes:
                self.roots(node, roots)
        else:
            if self.nodes[nodes]["root"]:
                roots.append(nodes)
            for i in nx.algorithms.ancestors(self, nodes):
                if self.nodes[i]["root"]:
                    roots.append(i)

        return list(set(roots))

    def clone(self, node, input_givens):

        if isinstance(node, t.Constant):
            return node.value
        # first simple case, if node is already in givens
        if node in input_givens:
            return input_givens[node]
        elif len(input_givens) == 0:
            return node
        elif isinstance(node, t.OpItem):
            parent = list(self.predecessors(node))[0]
            index = int(self[parent][node]["name"].split("parent_index")[1])
            return self.clone(parent, input_givens)[index]
        givens = input_givens.copy()

        # first we remove the redundant givens
        ancestors = nx.algorithms.ancestors(self, node)
        ancestors = set(ancestors).intersection(set(givens.keys()))

        # first is there is no more candidate, nothing todo
        if len(ancestors) == 0:
            return node

        # otherwise we remove the irrelevant ones
        for key in set(givens.keys()) - ancestors:
            givens.pop(key)

        if type(node) == list or type(node) == list or type(node) == t.Tuple:
            return [self.clone(n, givens) for n in node]

        # next we create a unique identifier. This will allow us to detect
        # in case this clone has already been created
        # this identifier is unique given a givens dictionnary
        names = [
            "{}{}->{}{}".format(m.scope, m.name, v.scope, v.name)
            for m, v in givens.items()
        ]
        names.sort()
        name = "_".join(names)

        # ToDo: work on that, it is not
        # good as it keeps in memory wrong
        # tensors that resutls from previous
        # call with different neighbours
        if 0:  # name in self._branches:
            return self._branches[name]

        args, kwargs = self.get_args_kwargs(node, evaluate=False)
        new_args = [self.clone(n, givens) for n in args if not isinstance(n, t.Seed)]
        new_kwargs = {name: self.clone(n, givens) for name, n in kwargs.items()}

        fun = self.get_node_attribute(node, "jax_function")
        self._branches[name] = symjax._fn_to_op[fun](*new_args, **new_kwargs)
        return self._branches[name]

    def get_node_attribute(self, node, attr):
        return nx.get_node_attributes(self, attr)[node]

    def get(self, item, tracker=None, frozen=True):
        """
        Example:

            >>> import symjax.tensor as T
            >>> w = T.ones(10).sum()+4
            >>> v = T.Variable(1., name='v', dtype='float32')
            >>> T.get(w+v)
            DeviceArray(15., dtype=float32)
        """
        value = self._get(item, tracker, frozen)
        if isinstance(value, jax.interpreters.xla.DeviceArray):
            return jax.device_get(value)

        return value

    def _get(self, item, tracker, frozen):
        """
        Example:

            >>> import symjax.tensor as T
            >>> w = T.ones(10).sum()+4
            >>> v = T.Variable(1., name='v', dtype='float32')
            >>> T.get(w+v)
            DeviceArray(15., dtype=float32)
        """
        if isinstance(item, t.Shape):
            args, kwargs = self.get_args_kwargs(item, evaluate=False)
            assert len(args) == 1
            assert len(kwargs) == 0
            return self.get_shape_dtype(args[0]).shape

        if tracker is None:
            tracker = {}

        if not t.isvar(item):
            return item

        if type(item) == list:
            return [self.get(i, tracker, frozen=frozen) for i in item]

        elif type(item) == tuple:
            return tuple([self.get(i, tracker, frozen=frozen) for i in item])

        elif item in tracker:
            return tracker[item]

        elif isinstance(item, t.Placeholder):
            if item not in tracker:
                raise ValueError(" no value given for placeholder {}".format(item))

        elif isinstance(item, t.Variable) or type(item) == t.Constant:
            tracker[item] = item.value
            if not frozen and isinstance(item, t.Seed):
                item.update()
            return tracker[item]

        elif type(item) == t.OpItem:

            return self.get(item.parent, tracker)[item.parent_index]

        elif isinstance(item, t.Op):

            # first get the actual parents nodes (aka inputs to the function)
            args, kwargs = self.get_args_kwargs(item, tracker, frozen=frozen)
            tracker[item] = self.nodes[item]["jax_function"](*args, **kwargs)

            return tracker[item]

        else:
            return item

    def get_shape_dtype(self, item):
        """
        Example:

            >>> import symjax.tensor as T
            >>> w = T.ones(10).sum()+4
            >>> v = T.Variable(1., name='v', dtype='float32')
            >>> T.get(w+v)
            DeviceArray(15., dtype=float32)
        """

        if "_shape" in self.nodes[item]:
            return jax.ShapeDtypeStruct(
                self.nodes[item]["_shape"], self.nodes[item]["dtype"]
            )

        elif type(item) == t.OpItem:

            return self.get_shape_dtype(item.parent)[item.parent_index]

        elif isinstance(item, t.Op):

            # first get the actual parents nodes (aka inputs to the function)
            args, kwargs = self.get_args_kwargs(item, evaluate=False)
            return t.get_output_tree(self.nodes[item]["jax_function"], *args, **kwargs)

    def get_args_kwargs(self, node, tracker=None, evaluate=True, frozen=True):
        if evaluate:
            assert tracker is not None
            all_args = {
                self[parent][node]["name"]: self.get(parent, tracker, frozen=frozen)
                for parent in self.predecessors(node)
            }
        else:
            all_args = {
                self[parent][node]["name"]: parent for parent in self.predecessors(node)
            }
        # now we inspect if there are duplicate args
        for arg in list(all_args.keys()):
            if "+" in arg:
                items = arg.split("+")
                for item in items:
                    all_args.update({item: all_args[arg]})
                del all_args[arg]

        arg_names = [name for name in all_args.keys() if "arg" == name[:3]]
        arg_names = sorted(arg_names, key=natural_key)

        args = [all_args[name] for name in arg_names]
        for name in arg_names:
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
        ops = [n for n in self.nodes if type(n) in [t.Op, t.MultiOutputOp]]
        return ops

    @property
    def updates(self):
        return self._updates

    @property
    def other_nodes(self):
        return list(
            set(self.nodes)
            - (
                set(self.ops).union(
                    set(self.placeholders).union(set(self.variables(None)))
                )
            )
        )


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

    def __init__(self, name, graph=None, reattach=False):
        """Constructor."""
        if graph is None:
            graph = current_graph()
        self.graph = graph
        # assert len(name)
        # assert "_" != name[-1]
        self.name = name
        self.full_name = None

    def __enter__(self):
        """Set global variables."""

        current = self.graph._current_scope
        cpt = 0
        if current + self.name + "/" in self.graph._scopes_history:
            while (
                current + self.name + "_{}/".format(cpt) in self.graph._scopes_history
            ):
                cpt += 1
            self.name += "_{}".format(cpt)

        if len(self.name):
            self.graph._current_scope += self.name + "/"
            self.full_name = self.graph._current_scope
        else:
            self.full_name = self.graph._current_scope
        self.graph._scopes.append(self)
        self.graph._scopes_history.append(self.full_name)
        return self

    def __exit__(self, *a):
        """Delete globals."""
        self.graph._scopes.pop(-1)
        if self.graph._scopes[-1].full_name is not None:
            self.graph._current_scope = self.graph._scopes[-1].full_name
        else:
            self.graph._current_scope = "/"

    def save_variables(self, path):
        """Save graph."""
        if ".npz" != path[:-4]:
            path += ".npz"
        numpy.savez(
            path,
            **dict([(v.name, symjax.tensor.get(v)) for v in self.variables]),
        )

    def variables(self, trainable=True):
        return get_variables(scope=self.full_name, trainable=trainable)

    @property
    def ops(self):
        return get_ops(scope=self.full_name)

    @property
    def placeholders(self):
        return get_placeholders(scope=self.full_name)

    def load_variables(self, path):
        """Load graph."""

        data = numpy.load(path)
        for name, value in data.items():
            self.variable[name].update(value)

    def reset(self):

        for var in self.variables:
            var.reset()

    def _get_name_scope(self, name, tensor):

        if self.full_name is None:
            self.__enter__()

        if isinstance(tensor, symjax.tensor.Placeholder):
            nodes = self.graph.placeholders
        elif isinstance(tensor, symjax.tensor.Variable):
            nodes = self.graph.variables(None)
        elif isinstance(tensor, symjax.tensor.Op):
            nodes = self.graph.ops
        else:
            nodes = self.graph.other_nodes

        names = [m.scope + m.name for m in nodes if hasattr(m, "name")]
        scope = self.full_name
        test_name = self.full_name + name

        if test_name not in names:
            return name, scope

        count = 1
        while test_name + "_" + str(count) in names:
            count += 1

        return name + "_" + str(count), scope


def reset_variables(name="*", scope="*", trainable=None):
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

    variables = get_variables(name=name, scope=scope, trainable=trainable)
    for var in variables:
        var.reset()


def save_variables(
    path_or_file,
    name="*",
    scope="*",
    trainable=None,
):
    """saves the graph variables.

    The saving is done via ``numpy.savez`` for fast and compressed storage.

    Parameters:
    -----------

    path_or_file: str or file
        the path and name of the file to save the variables in or an
        open file object

    name: str (optional)
        the name string that the variables to save must match

    scope: str (optional)
        the scope name string that the variables to save must match

    trainable: bool or None
        the option of the variables to save (``True``, ``False`` or ``None``)


    """
    if type(path_or_file) == str:
        if path_or_file[-4:] != ".npz":
            path_or_file += ".npz"
    variables = get_variables(name, scope, trainable)
    numpy.savez(
        path_or_file,
        **dict(
            [
                (
                    v.scope + v.name,
                    symjax.tensor.get(v),
                )
                for v in variables
            ]
        ),
    )


def load_variables(path_or_file, name="*", scope="*", trainable=None):
    """loads the graph variables.

    The loading is done via ``numpy.savez`` for fast and compressed storage.

    Parameters:
    -----------

    path_or_file: str or file
        the path and name of the file to load the variables from or an
        open file object

    name: str (optional)
        the name string that the variables to load must match

    scope: str (optional)
        the scope name string that the variables to load must match

    trainable: bool or None
        the option of the variables to save (``True``, ``False`` or ``None``)


    """

    if type(path_or_file) == str:
        if path_or_file[-4:] != ".npz":
            path_or_file += ".npz"

    variables = get_variables(name, scope, trainable=trainable)
    data = numpy.load(path_or_file)
    for var in variables:
        name_in_file = var.scope + var.name
        if name_in_file not in data:
            raise Warning("{} not in loaded file".format(name_in_file))
        var.update(data[name_in_file])


def get_variables(name="*", scope="/", trainable=True):
    matched = current_graph().variables(trainable)
    output = []
    for m in matched:
        if len(fnmatch.filter([m.name], name)) and len(
            fnmatch.filter([m.scope], scope + "*")
        ):
            output.append(m)
    return output


def get_placeholders(name="*", scope="/"):
    """
    Same as symjax.variable but for placeholders
    """
    matched = current_graph().placeholders
    output = []
    for m in matched:
        if len(fnmatch.filter([m.name], name)) and len(
            fnmatch.filter([m.scope], scope + "*")
        ):
            output.append(m)
    return output


def get_ops(name="*", scope="/"):
    """
    Same as symjax.variable but for ops
    """
    matched = current_graph().ops
    output = []
    for m in matched:
        if len(fnmatch.filter([m.name], name)) and len(
            fnmatch.filter([m.scope], scope + "*")
        ):
            output.append(m)
    return output


def get_updates(name="*", scope="/", variables=None):
    """
    Same as symjax.variable but for ops
    """
    matched = current_graph().updates
    output = {}

    if variables is not None:
        for v in variables:
            if v in matched:
                output[v] = matched[v]
        return output

    for var, update in matched.items():
        if len(fnmatch.filter([update.name], name)) and len(
            fnmatch.filter([update.scope], scope + "*")
        ):
            output[var] = update
    return output


def add_updates(updates):
    current_scope().add_updates(updates)


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
    if numpy.prod(scalar.shape.get()) != 1:
        raise RuntimeError("the variable to differentiate is not a scalar")
    if not isinstance(scalar, t.Tensor):
        raise RuntimeError("the variable used in gradients should be a Tensor type")

    if scalar.shape != ():
        scalar = scalar.sum()
    if isinstance(variables, t.Tensor):
        input_variables = [variables]
        input_list = False
    else:
        input_variables = variables.copy()
        input_list = True

    # get the argnum of the variables that we differentiate one
    argnums = list(range(len(input_variables)))

    # get all the roots of the scalar, this is needed as otherwise they are not
    # as the input of the gradient function and thus a change of
    # their value will not change the gradient computation, we also ensure
    # uniqueness
    input_variables += [
        i for i in current_graph().roots(scalar) if i not in input_variables
    ]

    # create a dummy function that is needed for jax to compute a gradient func
    # this function is the one that builds the graph of computation from all
    # roots
    # to the scalar varible s.t. automatic diffenrentiation can be applied

    def fn(*args):
        return current_graph().get(scalar, dict(zip(input_variables, list(args))))

    # now we obtain the grad function. In fact, Jax returns a function that,
    # when it is called, returns the gradient values, this function is then
    # used to generate the Tuple of symbolic variables
    grad_fn = jax.grad(fn, argnums)
    wrap_fn = t.jax_wrap(grad_fn)
    if input_list:
        return wrap_fn(*input_variables)
    else:
        return wrap_fn(*input_variables)[0]


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
        default_value=None,
        frozen=False
    ):
        """Initialize."""
        # check the given updates (if any) and ensure that they only
        # update Variable objects
        if updates is None:
            updates = {}
        else:
            current_graph().add_updates(updates)

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
        self.updates_keys = list(updates.keys())
        self.updates_values = list(updates.values())
        self.classargs = classargs
        self.outputs = outputs

        outs = self.updates_values
        outs += [outputs] if isinstance(outputs, t.Tensor) else outputs

        self.all_roots = set(symjax.current_graph().roots(outs))

        # check the function inputs, they must be at least contain all the
        # placeholders needed to compute the outputs values
        placeholders_in_root = [
            x for x in self.all_roots if isinstance(x, t.Placeholder)
        ]

        # check for
        non_givens = set(placeholders_in_root) - set(self.classargs)
        if len(non_givens) > 0:
            raise RuntimeError(
                """\
                Missing placeholders from the function inputs...\n\
                \t...Givens are: {}\n\
                \t...Missings are: {}""".format(
                    placeholders_in_root, non_givens
                )
            )

        # the roots are made of variables, random tensors, placeholders. We
        # already ensured that all placeholders are given as inputs to the
        # function. Now we must ensure that the other ones will also be given
        # as inputs to not be treated as constants by jax.
        # we also remove update keys because we will expicitly feed them
        self.extra_inputs = set(self.all_roots) - set(self.classargs).union(
            self.updates_keys
        )

        self.extra_inputs = list(self.extra_inputs)
        allargs = list(self.classargs) + self.updates_keys + self.extra_inputs

        def to_jit(*jitargs, seed):

            feed_dict = dict(zip(allargs, jitargs))
            feed_dict.update({"rng": seed})
            outputs = [self.outputs, self.updates_values]
            return symjax.current_graph().get(outputs, feed_dict)

        # take care of the presence of -1 dimensions
        to_vectorize = [0 if 0 in symjax.tensor.get(a.shape) else None for a in allargs]

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
                    if (
                        fnarg.shape != classarg.shape.get()
                        and 0 not in classarg.shape.get()
                    ):
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

            if not frozen:
                for i in self.extra_inputs:
                    if isinstance(i, t.Seed):
                        i.update()

            return jitoutputs

        self.meta = meta

    def __call__(self, *args, rng=None, device_get=True):
        """Callable fn."""
        # in the presence of RandomTensor(s) in the graph, we keep track of the
        # number of functions calls to keep accumulating the PRNGKey of the jax
        # key, otherwise each function call returns the same realisation

        if rng is None:
            rng = random._seed
            random._seed += 1

        outputs = self.meta(*args, rng=rng)

        if device_get:
            outputs = jax.device_get(outputs)
        return outputs
