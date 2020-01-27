import jax
import jax.numpy as np
from . import tensor

from jax import jacfwd, jacrev


def gradients(scalar, deps, aggregation=tensor.sum):

    if scalar.shape != ():
        scalar = aggregation(scalar)

    # get all the roots, this is needed as otherwise they are not
    # as the input of the gradient function and thus a change of
    # their value will not change the gradient computation
    # now we check if we have to differentiate w.r.t a non root variable
    to_add = list(set(scalar.roots) - set(deps))
    all_roots = scalar.roots + to_add

    # get the argnum (index of the function input that will have to be
    # differentiated)
    argnums = [all_roots.index(dep) for dep in deps]

    # create a dummy function that is needed for jax to compute a gradient func
    def fn(*args):
        return scalar.get(dict(zip(all_roots, list(args))))

    grad_fn = jax.grad(fn, argnums)
    return tensor.Tuple(grad_fn, args=all_roots)


def jacobian_forward(scalar, deps):
    # good for tall J
    # get all the roots, this is needed as otherwise they are not
    # as the input of the gradient function and thus a change of
    # their value will not change the gradient computation
    all_roots = scalar.roots

    # now we check if we have to differentiate w.r.t a non root variable
    to_add = [dep for dep in deps if dep not in all_roots]
    all_roots += to_add

    # get the argnum (index of the function input that will have to be
    # differentiated)
    argnums = [i for i, arg in enumerate(all_roots) if arg in deps]

    # create a dummy function that is needed for jax to compute a gradient func
    def fn(*args):
        return scalar.get(dict(zip(all_roots, list(args))))

    grad_fn = jax.jacfwd(fn, argnums)
    return tensor.Tuple(grad_fn, args=all_roots)


def jacobian_backward(scalar, deps):
    # good for wide J
    # get all the roots, this is needed as otherwise they are not
    # as the input of the gradient function and thus a change of
    # their value will not change the gradient computation
    all_roots = scalar.roots

    # now we check if we have to differentiate w.r.t a non root variable
    to_add = [dep for dep in deps if dep not in all_roots]
    all_roots += to_add

    # get the argnum (index of the function input that will have to be
    # differentiated
    argnums = [i for i, arg in enumerate(all_roots) if arg in deps]

    # create a dummy function that is needed for jax to compute a gradient func
    def fn(*args):
        return scalar.get(dict(zip(all_roots, list(args))))

    grad_fn = jax.jacrev(fn, argnums)
    return tensor.Tuple(grad_fn, args=all_roots)


class function:

    def __init__(self, *classargs, outputs=[], updates=None, device=None,
                 backend=None, default_value=None):

        # check the given updates (if any) and ensure that they only
        # update Variable objects
        if updates is None:
            updates = {}
        for update in updates.keys():
            if not isinstance(update, tensor.Variable):
                raise RuntimeError(
                    "{} is not a Variable, it can not be updated".format(update))

#        # check the function inputs, they must be at least contain all the
#        # placeholders needed to compute the outputs values
#        placeholders = []
#        for output in outputs:
#            placeholders += filter(lambda x: isinstance(x, tensor.Placeholder),
#                                   output.roots)
#        placeholders = list(set(placeholders))
#        for placeholder in placeholders:
#            if placeholder not in classargs:
#                raise RuntimeError(
#                    "Placeholder {} was not given as function input".format(placeholder))

        # create the function that will take the inputs and return the update
        # values (if any) and the outputs, this function is jit compiled for
        # performances, we also add the roots/updates as hidden inputs

        # gather all roots
        all_roots = set(sum([output.roots for output in outputs], []))

        # ensure that not variable being update is also an input to the graph
        assert len(set(updates.keys()).intersection(set(classargs))) == 0

        # we need to add as input of this functin any variable not just the ones
        # being updates, otherwise they are treated as constant by jax compiled
        # function, we thus now search which variables do we need to add
        # we also don't optimize by searching if the path to some variables
        # was cut by a given input as it won't happen in general

        update_inputs = list(updates.keys())
        # retreive the variables that are in the roots
        all_var_roots = filter(
            lambda x: isinstance(x, tensor.Variable), list(all_roots))
        hidden_inputs = list(set(all_var_roots) - set(update_inputs))
#        for output in outputs:
#            only_vars = filter(
#                lambda x: isinstance(x, tensor.Variable), output.roots)
#            for variable in only_vars:
#                if variable not in hidden_inputs:
#                    hidden_inputs.append(variable)

        def jitfn(*jitargs, rng):

            allargs = list(classargs) + update_inputs + hidden_inputs
            kwargs = dict(zip(allargs, jitargs))
            kwargs.update({'rng': rng})

            # compute the outputs
            jit_outputs = [output.get(kwargs) for output in outputs]

            # compute the values of the updates
            jit_updates = [output.get(kwargs) for output in updates.values()]
            return jit_outputs, jit_updates

        # we compile our underlying function
        jitfn = jax.jit(jitfn, device=device, backend=backend)
        self.jitfn = jitfn

        # define the frontend function that takes as input the inputs variables
        # and internally compute and update the variables if any
        def meta(*fnargs, rng):

            # ensure that the number of arguments is correct
            assert len(fnargs) == len(classargs)

            # get the addition inputs to the function (the values to be
            # updated)
            hidden_values = [
                var.value for var in update_inputs + hidden_inputs]

            # retreive the function outputs and updated values and apply them
            fn_outputs, fn_updates = self.jitfn(*fnargs, *hidden_values, rng=rng)
            for key, update in zip(update_inputs, fn_updates):
                key.value = update
            return fn_outputs

        self.meta = meta

    def __call__(self, *args, rng=None):

        # in the presence of RandomTensor(s) in the graph, we keep track of the
        # number of functions calls to keep accumulating the PRNGKey of the jax
        # key, otherwise each function call returns the same realisation
        if rng is None:
            if '_rng' not in globals():
                globals()['_rng'] = 0
            globals()['_rng'] += 1
            rng = globals()['_rng']

        return self.meta(*args, rng=rng)
