import jax
import jax.numpy as np
import numpy as NP
from . import tensor

def gradients(scalar, params):
    assert scalar.shape == ()
    deps = scalar.all_dependencies
    argnums = [i for i, dep in enumerate(deps) if dep in params]
    def fn(*args):
        scalar.reset_value(True)
        return scalar.get(dict(zip(deps, list(args))))
    shapes = [param.shape for param in params]
    dtypes = [param.dtype for param in params]
    grad_fn = jax.grad(fn, argnums)
    return tensor.List(grad_fn, shapes, dtypes, args=deps)


class function:
    def __init__(self, *classargs, outputs=[], updates={}, device=None):

        # ensure that we only aim at updating variables
        for update in updates.keys():
            assert isinstance(update, tensor.Variable)

        # now create the function that will take the inputs and return
        # the update values (if any) and the outputs, this function is the one that
        # will be jit compiled for performances
        def jitfn(*jitargs, rng):

            # the args are made of the actual inputs of the final function as
            # well as the values to be updated, we thus concatenate the latter to
            # the former and then convert to a dict
            allargs = list(classargs) + list(updates.keys())
            kwargs = dict(zip(allargs, jitargs))
            kwargs.update({'rng': rng})

            # compute the outputs
            jit_outputs = [output.get(kwargs) for output in outputs]

            # compute the values of the updates
            jit_updates = [output.get(kwargs) for output in updates.values()]
            return jit_outputs, jit_updates

        # we compile our underlying function
        jitfn = jax.jit(jitfn, device_assignment=device)
        self.jitfn = jitfn

        # now define the actual user-function that will only take as input the
        # inputs variables and internally also compute and update the variables
        # if any, that are in updates
        def meta(*fnargs, rng):

            # ensure that the number of arguments is correct
            assert len(fnargs) == len(classargs)

            # get the addition inputs to the function (the values to be updated)
            variables = list(updates.keys())
            update_args = [var.value for var in variables]

            # retreive the function outputs and updated values and apply them
            fn_outputs, fn_updates = self.jitfn(*fnargs, *update_args, rng=rng)
            for key, update in zip(variables, fn_updates):
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
