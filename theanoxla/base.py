import jax
import jax.numpy as np
import numpy as NP
from . import tensor

def gradients(scalar, params):
    assert scalar.shape == ()
    deps = scalar.all_dependencies
    argnums = [i for i, dep in enumerate(deps) if dep in params]
    def fn(*args):
        return scalar.get(dict([(dep,arg) for arg, dep in zip(args, deps)]),
                          force=True)
    shapes = [param.shape for param in params]
    dtypes = [param.dtype for param in params]
    grad_fn = jax.grad(fn, argnums)
    return tensor.List(grad_fn, shapes, dtypes, args=deps)


def function(*args, outputs=[], updates={}, device=None):

    # ensure that we only aim at updating variables
    for update in updates.keys():
        assert isinstance(update, tensor.Variable)

    # now create the function that will take the inputs and return
    # the update values (if any) and the outputs, this function is the one that
    # will be jit compiled for performances
    def jitfn(*args1):

        # the args are made of the actual inputs of the final function as
        # well as the values to be updated, we thus concatenate the latter to
        # the former and then convert to a dict
        _args = list(args) + list(updates.keys())
        kwargs = dict([(key, value) for key, value in zip(_args, args1)])

        # reset the values. This is needed as we do not force the value
        # computation below, and values might already have been computed with
        # some tracer to evaluate shape etc ... hence we force a full
        # graph evaluation again
        for output in outputs:
            output.reset_value(True)
        for output in updates.values():
            output.reset_value(True)

        # compute the outputs
        fn_outputs = [output.get(kwargs) for output in outputs]

        # compute the values of the updates
        fn_updates = [output.get(kwargs) for output in updates.values()]
        return fn_outputs, fn_updates

    # we compile our underlying function
    jitfn = jax.jit(jitfn, device_assignment=device)

    # now define the actual user-function that will only take as input the
    # inputs variables and internally also compute and update the variables
    # if any, that are in updates
    def meta(*args2):
        assert len(args2) == len(args)
        variables = list(updates.keys())
        # retreive the function outputs and updated values
        outputs2, updates2 = jitfn(*args2, *[var.value for var in variables])
        # update the variables
        for key, update in zip(variables, updates2):
            key.value = update
        return outputs2

    return meta
