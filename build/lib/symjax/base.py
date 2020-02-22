import jax
import jax.numpy as np
from . import tensor as t
from jax import jacfwd, jacrev


def gradients(scalar, variables):
    """computes the gradients of a scalar w.r.t to a given list of variables.

    Arguments
    ---------

        scalar: Tensor
            the variable to differentiate

        variables: List or Tuple
            the variables used to compute the derivative.

    Returns
    -------

        gradients: Tuple
            the sequency of gradients ordered as given in the input variables
    """


    if scalar.shape != ():
        raise RuntimeError("the variable to differentiate is not a scalar")
    elif not isinstance(scalar, t.Tensor):
        raise RuntimeError(
            "the variable used in gradients should be a Tensor type")


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
    # this function is the one that builds the graph of computation from all roots
    # to the scalar varible s.t. automatic diffenrentiation can be applied
    def fn(*args):
        return scalar.get(dict(zip(all_roots, list(args))))

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
    """computes the jacobians of a tensor w.r.t to a given list of variables.
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
    """

    # get all the roots of the scalar, this is needed as otherwise they are not
    # as the input of the gradient function and thus a change of
    # their value will not change the gradient computation, we also ensure
    # uniqueness
    all_roots = list(set(tensor.roots + variables))

    # get the argnum of the variables that we differentiate one
    argnums = [all_roots.index(var) for var in variables]

    # create a dummy function that is needed for jax to compute a gradient func
    # this function is the one that builds the graph of computation from all roots
    # to the scalar varible s.t. automatic diffenrentiation can be applied
    def fn(*args):
        return tensor.get(dict(zip(all_roots, list(args))))

    # now we obtain the jacobian function. In fact, Jax returns a function that
    # when it is called, returns the jacobian values, this function is then
    # used to generate the Tuple of symbolic variables
    if mode == 'forward':
        jacob_fn = jacfwd(fn, argnums)
    elif mode == 'backward':
        jacob_fn = jacrev(fn, argnums)
    else:
        raise RuntimeError(
            "given mode {} is not recognized, use forward or backward".format(mode))
    wrap_fn = t.jax_wrap(jacob_fn, False)
    return wrap_fn(*all_roots)


class function:

    """generates a user function that compiles a computational graph
    based on given inputs, outputs and update policy of variables. This
    function internally jit compile the underlying jax computational
    graph for performances and thus should be favored to the get
    method of tensors.

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

        device: ??
            ??

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

        >>> import jaxonn
        >>> import jaxonn.tensor as T
        >>> x = T.ones((4, 4))
        >>> xs = x.sum() + 1
        >>> f = jaxonn.function(outputs=xs)
        >>> print(f()) # returns 17

        >>> w = T.Variable(0., name='w')
        >>> increment = jaxonn.function(updates={w: w + 1})
        >>> for i in range(10):
        >>>     increment()
        >>> print(w.value) # returns 10

    """

    def __init__(self, *classargs, outputs=[], updates=None, device=None,
                 backend=None, default_value=None):

        # check the given updates (if any) and ensure that they only
        # update Variable objects
        if updates is None:
            updates = {}

        for update in updates.keys():
            if not isinstance(update, t.Variable):
                raise RuntimeError(
                    "{} is not a Variable, it can not be updated".format(update))

        # ensure that all inputs are actual placeholders or variables
        for arg in classargs:
            if not isinstance(arg, t.Tensor):
                raise RuntimeError(
                    "{} is not a Tensor type. Only tensor types can be function inputs".format(arg))
        # gather all roots, they need to be explicit as inputs of the
        # underlying functions otherwise they are treated as constants
        # and any change in their value will not appear when running the
        # function
        all_roots = list(updates.values())
        all_roots += [outputs] if isinstance(outputs, t.Tensor) else outputs
        self.all_roots = set(t.getroots(all_roots))
        self.classargs = classargs
        self.outputs = outputs
        items = list(updates.items())
        self.updates_keys = [item[0] for item in items]
        self.updates_values = [item[1] for item in items]

        # check the function inputs, they must be at least contain all the
        # placeholders needed to compute the outputs values
        placeholders = filter(
            lambda x: isinstance(x, t.Placeholder),
            self.all_roots)
        non_givens = set(placeholders) - set(self.classargs)
        if len(non_givens) > 0:
            raise RuntimeError(
                "Missing placeholders form the function inputs: {}".format(non_givens))

        # the roots are made of variables, random tensors, placeholders. We
        # already ensured that all placeholders are given as inputs to the
        # function. Now we must ensure that the other ones will also be given
        # as inputs to not be treated as constants by jax.
        self.extra_inputs = set(self.all_roots)
        self.extra_inputs -= set(self.classargs)
        self.extra_inputs -= set(self.updates_keys)
        self.extra_inputs = list(self.extra_inputs)

        def jitfn(*jitargs):
            allargs = list(self.classargs) + self.updates_keys +\
                                            self.extra_inputs
            kwargs = dict(zip(allargs, jitargs))

            # compute the outputs
            jit_outputs = t.get(self.outputs, kwargs)

            # compute the values of the updates
            jit_updates = t.get(self.updates_values, kwargs)

            return jit_outputs, jit_updates

        # we compile our underlying function using jit for performances
        self.jitfn = jax.jit(jitfn, device=device, backend=backend)

        # define the frontend function that takes as input the inputs variables
        # and internally compute and update the variables from updates if any
        def meta(*fnargs, rng):

            # ensure that the number of arguments is correct
            assert len(fnargs) == len(self.classargs)
            for fnarg, classarg in zip(fnargs, self.classargs):
                if hasattr(fnarg, 'shape'):
                    if fnarg.shape != classarg.shape:
                        raise RuntimeError("wrong input given for {}".format(classarg))

            # get the addition inputs to the function (the values to be
            # updated)
            extra_values = t.get(self.updates_keys + self.extra_inputs,
                                 {'rng': rng})

            # retreive the function outputs, updated values and apply them
            jitoutputs, jitupdates = self.jitfn(*fnargs, *extra_values)
            for key, update in zip(self.updates_keys, jitupdates):
                key.value = update
            return jitoutputs

        self.meta = meta

    def __call__(self, *args, rng=None):

        # in the presence of RandomTensor(s) in the graph, we keep track of the
        # number of functions calls to keep accumulating the PRNGKey of the jax
        # key, otherwise each function call returns the same realisation
        if rng is None:
            if '_rng' not in globals():
                globals()['_rng'] = 0
            globals()['_rng'] += 1
            _rng = globals()['_rng']
        else:
            _rng = rng
        return self.meta(*args, rng=_rng)
