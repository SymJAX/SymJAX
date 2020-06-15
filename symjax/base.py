#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base of symjax."""

import fnmatch
import warnings

import jax
import numpy
from jax import jacfwd, jacrev

import symjax
from symjax import tensor as t
from symjax.tensor import random


def current_graph():
    """Current graph."""
    assert len(symjax._current_graph)
    return symjax._current_graph[-1]


class Graph:
    """Graph."""

    def __init__(self, name, seed=None):

        """Constructor."""

        self.name = name
        self.full_name = None

    def __enter__(self):
        """Set global variables."""
        if len(self.name):
            symjax._current_scope += self.name + '/'
        self.full_name = symjax._current_scope
        symjax._current_graph.append(self)

    def __exit__(self, *a):
        """Delete globals."""
        symjax._current_graph.pop(-1)
        if symjax._current_graph[-1].full_name is not None:
            symjax._current_scope = symjax._current_graph[-1].full_name
        else:
            symjax._current_scope = '/'

    def save_variables(self, path):
        """Save graph."""
        numpy.savez(path, **dict(
            [(v.name, symjax.tensor.get(v)) for v in self.variables]))

    @property
    def variables(self):
        return get_variables(self.full_name + '*')

    def variable(self, name):

        # check if the name for given relative or full
        if '/' not in name:
            full_name = self.full_name + name
        else:
            full_name = name

        if full_name in symjax._variables:
            return symjax._variables[full_name]
        else:
            RuntimeError('Variable {name} not in graph {self.full_name}')

    def load_variables(self, path):

        """Load graph."""

        data = numpy.load(path)
        for name, value in data.items():
            self.variable[name].update(value)

    def reset(self):

        for var in self.variables:
            var.reset()

    def add(self, tensor):

        # fake graph entrance if not used by user
        if self.full_name is None:
            self.__enter__()

        tensor.scope = self.full_name
        name = self.full_name + tensor.name
        if isinstance(tensor, symjax.tensor.Placeholder):
            names = symjax._placeholders
        elif isinstance(tensor, symjax.tensor.Variable):
            names = symjax._variables
        else:
            names = symjax._ops
        # print('in add', names)
        if name not in names.keys():
            names[name] = tensor
            return

        count = 1
        while True:
            if name + '_' + str(count) in names.keys():
                count += 1
            else:
                break
        names[name + '_' + str(count)] = tensor
        tensor._set_name(tensor.name + '_' + str(count))


def reset_variables(name='*', trainable=None):
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
        >>> import logging
        >>> w = symjax.tensor.Variable(1., name='w', dtype='float32')
        >>> x = symjax.tensor.Variable(2., name='x', dtype='float32')
        >>> f = symjax.function(outputs=[w, x], updates={w:w + 1,x:x + 1})
        >>> for i in range(10):
        ...    f()
        >>> # reset only the w variable
        >>> symjax.reset_variables('w')
        >>> # reset all variables
        >>> symjax.reset_variables('*')

    """

    matched = fnmatch.filter(symjax._variables.keys(), name)
    for m in matched:
        if trainable is not None:
            if symjax._variables[m].trainable != trainable:
                continue
        symjax._variables[m].reset()


def save_variables(name, path):
    """Save graph."""
    matched = fnmatch.filter(symjax._variables.keys(), name)
    numpy.savez(path, **dict(
        [(symjax._variables[v].scope + symjax._variables[v].name,
          symjax.tensor.get(symjax._variables[v])) for v in
         matched]))


def load_variables(name, path_or_file, scope_mapping=None):
    """Load graph."""

    if type(path_or_file) == str:
        if path_or_file[-4:] != '.npz':
            path_or_file += '.npz'

    scope_mapping = scope_mapping or {}

    matched = fnmatch.filter(symjax._variables.keys(), name)
    data = numpy.load(path_or_file)
    for name in matched:
        if symjax._variables[name].scope in scope_mapping:
            name_in_file = scope_mapping[symjax._variables[name].scope] + '/' + \
                           symjax._variables[name].name
        else:
            name_in_file = name
        if name_in_file not in data:
            raise Warning('{} not in loaded file'.format(name_in_file))
        symjax._variables[name].update(data[name_in_file])


def get_variables(name, trainable=None):
    matched = fnmatch.filter(symjax._variables.keys(), name)
    if trainable is not None:
        assert type(trainable) == bool
        matched = [m for m in matched
                   if symjax._variables[m].trainable == trainable]
    return [symjax._variables[m] for m in matched]


def get_placeholders(name):
    """
    Same as symjax.variable but for placeholders
    """

    matched = fnmatch.filter(symjax._placeholders.keys(), name)
    return [symjax._placeholders[m] for m in matched]


def get_ops(name):
    """
    Same as symjax.variable but for ops
    """

    matched = fnmatch.filter(symjax._ops.keys(), name)
    return [symjax._ops[m] for m in matched]


def get_updates(name):
    """
    Same as symjax.get_variables but for updates
    """

    matched = fnmatch.filter(symjax._updates.keys(), name)
    return [symjax._ops[m] for m in matched]


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
    """
    if numpy.prod(scalar.shape) != 1:
        raise RuntimeError("the variable to differentiate is not a scalar")
    if not isinstance(scalar, t.Tensor):
        raise RuntimeError(
            "the variable used in gradients should be a Tensor type")

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
    all_roots = list(set(scalar.roots + input_variables))

    # get the argnum of the variables that we differentiate one
    argnums = [all_roots.index(var) for var in input_variables]

    # create a dummy function that is needed for jax to compute a gradient func
    # this function is the one that builds the graph of computation from all
    # roots
    # to the scalar varible s.t. automatic diffenrentiation can be applied
    def fn(*args):
        return symjax.tensor.get(scalar, dict(zip(all_roots, list(args))))

    # now we obtain the grad function. In fact, Jax returns a function that,
    # when it is called, returns the gradient values, this function is then
    # used to generate the Tuple of symbolic variables
    grad_fn = jax.grad(fn, argnums)
    wrap_fn = t.jax_wrap(grad_fn, False)
    if input_list:
        return wrap_fn(*all_roots)
    else:
        return wrap_fn(*all_roots)[0]


def jacobians(tensor, variables, mode='forward'):
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
    if mode == 'forward':
        jacob_fn = jacfwd(fn, argnums)
    elif mode == 'backward':
        jacob_fn = jacrev(fn, argnums)
    else:
        raise RuntimeError(
            "mode {} not recognized, use forward or backward".format(mode))
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
        >>> print(f()) # returns 17

        >>> w = T.Variable(0., name='w')
        >>> increment = symjax.function(updates={w: w + 1})
        >>> for i in range(10):
        >>>     increment()
        >>> print(w.value) # returns 10

    """

    def __init__(self, *classargs, outputs=[], updates=None,  # noqa
                 device=None,
                 backend=None, default_value=None):
        """Initialize."""
        # check the given updates (if any) and ensure that they only
        # update Variable objects
        if updates is None:
            updates = {}

        for update in updates.keys():
            if not isinstance(update, t.Variable):
                raise RuntimeError(
                    "{} is not a Variable and cannot be updated".format(
                        update))

        # ensure that all inputs are actual placeholders or variables
        for arg in classargs:
            if not isinstance(arg, t.Tensor):
                raise RuntimeError(
                    "{} is not a Tensor type. Only tensor types can be" +
                    "function inputs".format(arg))

        # gather all roots, they need to be explicit as inputs of the
        # underlying functions otherwise they are treated as constants
        # and any change in their value will not appear when running the
        # function
        outs = list(updates.values())
        outs += [outputs] if isinstance(outputs, t.Tensor) else outputs
        self.all_roots = set(t.getroots(outs))
        self.classargs = classargs
        self.outputs = outputs
        items = list(updates.items())
        self.updates_keys = [item[0] for item in items]
        self.updates_values = [item[1] for item in items]
        for i in range(len(items)):
            if self.updates_keys[i].shape != self.updates_values[i].shape:
                warnings.warn(
                    'Variable and update {} {}'.format(
                        self.updates_keys[i],
                        self.updates_values[i]) +
                    "are not the same shape... attempting to reshape")
                self.updates_values[i] = t.reshape(self.updates_values[i],
                                                   self.updates_keys[i].shape)
            if self.updates_keys[i].dtype != self.updates_values[i].dtype:
                warnings.warn(
                    'Variable and update {} {}'.format(
                        self.updates_keys[i],
                        self.updates_values[i]) +
                    "are not the same dtype... attempting to cast")
                self.updates_values[i] = t.cast(self.updates_values[i],
                                                self.updates_keys[i].dtype)

        # check the function inputs, they must be at least contain all the
        # placeholders needed to compute the outputs values
        placeholders_in_root = filter(lambda x: isinstance(x, t.Placeholder),
                                      self.all_roots)

        # check for
        non_givens = set(placeholders_in_root) - set(self.classargs)
        if len(non_givens) > 0:
            raise RuntimeError(
                "Missing placeholders form the function inputs: {}".format(
                    non_givens))

        # the roots are made of variables, random tensors, placeholders. We
        # already ensured that all placeholders are given as inputs to the
        # function. Now we must ensure that the other ones will also be given
        # as inputs to not be treated as constants by jax.
        # we also remove update keys because we will expicitly feed them
        self.extra_inputs = set(self.all_roots) \
                            - (set(self.classargs).union(self.updates_keys))
        self.extra_inputs = list(self.extra_inputs)

        def to_jit(*jitargs, seed):
            allargs = list(
                self.classargs) + self.updates_keys + self.extra_inputs
            feed_dict = dict(zip(allargs, jitargs))  # [(m, {'base': v})
            #                        for m, v in zip(allargs, jitargs)])
            feed_dict.update({'rng': seed})
            return t.get([self.outputs, self.updates_values], feed_dict)

        # we compile our underlying function using jit for performances
        self.jited = jax.jit(to_jit, device=device, backend=backend)

        # define the frontend function that takes as input the inputs variables
        # and internally compute and update the variables from updates if any
        def meta(*fnargs, rng):

            # ensure that the number of arguments is correct
            assert len(fnargs) == len(self.classargs)
            for fnarg, classarg in zip(fnargs, self.classargs):
                if hasattr(fnarg, 'shape'):
                    if fnarg.shape != classarg.shape:
                        raise RuntimeError(
                            "wrong input given for {}".format(classarg) +
                            ", given is {}".format(fnarg) +
                            ", shape={}".format(fnarg.shape))

            # retreive the function outputs, updated values and apply them
            jited_add_inputs = t.get(self.updates_keys + self.extra_inputs,
                                     tracker={'rng': rng})
            jitoutputs, jitupdates = self.jited(*fnargs, *jited_add_inputs,
                                                seed=rng)
            for key, update in zip(self.updates_keys, jitupdates):
                key.update(update)
            if isinstance(jitoutputs, jax.interpreters.xla.DeviceArray):
                return jax.api.device_get(jitoutputs)
            else:
                npy_jitoutputs = [jax.api.device_get(arr) if isinstance(arr,
                                                                        jax.interpreters.xla.DeviceArray) else arr
                                  for arr in jitoutputs]
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
        return self.meta(*pargs, rng=rng)
